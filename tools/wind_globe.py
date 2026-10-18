# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "healpy", "cartopy", "pillow"]
# ///
"""The looping globe on the home page's wind card: generated stratospheric wind shaded on a turning
orthographic globe, glowing particles carried by it, and a few balloons drifting in it.

The wind is our generator's own output (Stage 1 -> Stage 2, HEALPix nside 256), not ERA5: the
13-hour block for 2023-07-10 00 UTC saved by balloon-env-dev/unicorn/eye_test.py, thinned to every
8th NEST pixel in ~/data/eye_jul10/fields_thin.npz on Unicorn. Pull just the generated field:

    ssh unicorn '~/envs/cbottle/bin/python -c "import numpy as np, os; z = np.load(os.path.expanduser(\\"~/data/eye_jul10/fields_thin.npz\\")); np.savez(os.path.expanduser(\\"~/tmp/site_globe/gen_b_thin.npz\\"), gen_b=z[\\"gen_b\\"], hours=z[\\"hours\\"])"'
    scp unicorn:tmp/site_globe/gen_b_thin.npz .

then

    uv run tools/wind_globe.py prepare gen_b_thin.npz wind_53hPa.npz          # HEALPix -> 0.5 deg lat-lon, one level
    uv run tools/wind_globe.py still   wind_53hPa.npz still.png --frame 0     # one frame, to judge the look
    uv run tools/wind_globe.py render  wind_53hPa.npz static/img/projects/wind.mp4 static/img/projects/wind.jpg

The loop is exact: the globe turns once, the wind's 13 hours play forward and back on a cosine, and
every particle and balloon is reborn on a schedule whose period divides the loop, so frame N is frame 0.
The look (speed-shaded globe in the dark, particle streaks, white balloons) follows Vinamr Jain's
globe renderer in zermelo; this is a separate implementation.
"""

import argparse
import subprocess
import sys

import numpy as np

EARTH_KM = 6371.0
LEVELS_HPA = {0: 53}  # channel pair 0 of the 18-level stack (model levels 49..66) is ~53 hPa

SIZE = 720
SS = 2  # supersampling: drawn at SIZE * SS, box-filtered down
RADIUS = 0.445  # globe radius as a fraction of the frame
TILT = -28.0  # the camera's latitude: low enough to see the southern winter vortex and the tropics
LON0 = 20.0  # the camera's longitude at frame 0
FPS = 24
FRAMES = 288  # 12 s: one turn of the globe
STEP_HOURS = 1.0  # hours the particles and balloons are carried per frame

VMAX = 60.0  # m/s at the top of the colour scale
GAMMA = 1.0

PARTICLES = 11000
LIFE = 36  # frames a particle lives; divides FRAMES
TRAIL = 16  # positions a streak is drawn through
BALLOONS = 8
BALLOON_LIFE = FRAMES // 2
BALLOON_TRAIL = 40

SPACE = np.array([4, 6, 12], np.float32)
# calm reads as deep navy, the jets as amber and cream
STOPS = [
    (0.00, "#060914"),
    (0.12, "#0b1633"),
    (0.26, "#12305f"),
    (0.42, "#1b5a92"),
    (0.56, "#2a8fae"),
    (0.68, "#6cbf8e"),
    (0.78, "#e0c050"),
    (0.88, "#f2832f"),
    (0.95, "#f4552a"),
    (1.00, "#ffe8c4"),
]
COAST = np.array([150, 172, 196], np.float32)
LAND = np.array([40, 52, 70], np.float32)
ATMOSPHERE = np.array([80, 160, 230], np.float32)
STREAK = np.array([255, 250, 238], np.float32)
TRACK = np.array([255, 222, 170], np.float32)
BALLOON = np.array([255, 255, 255], np.float32)


def hex_rgb(h):
    return np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], np.float32)


def lut(n=256):
    at = np.array([s for s, _ in STOPS])
    rgb = np.stack([hex_rgb(c) for _, c in STOPS])
    x = np.linspace(0, 1, n)
    return np.stack([np.interp(x, at, rgb[:, k]) for k in range(3)], axis=1).astype(np.float32)


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def latlon(p):
    return np.degrees(np.arcsin(np.clip(p[..., 2], -1, 1))), np.degrees(np.arctan2(p[..., 1], p[..., 0]))


# ---------------------------------------------------------------------------------------------- prepare


def prepare(src, out, level):
    """HEALPix (every 8th NEST pixel at nside 256) to a 0.5 deg grid, u and v at one level, all hours"""
    import healpy as hp
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import Delaunay

    z = np.load(src)
    g = z["gen_b"]
    lon, lat = hp.pix2ang(256, np.arange(0, 12 * 256 * 256, 12 * 256 * 256 // g.shape[-1]), nest=True, lonlat=True)
    lon = (lon + 180) % 360 - 180
    # pad across the dateline so the triangulation closes the seam
    east, west = lon > 150, lon < -150
    plon = np.concatenate([lon, lon[west] + 360, lon[east] - 360])
    plat = np.concatenate([lat, lat[west], lat[east]])
    pick = np.concatenate([np.arange(lon.size), np.flatnonzero(west), np.flatnonzero(east)])
    tri = Delaunay(np.column_stack([plon, plat]))
    glat = np.arange(-90, 90.01, 0.5)
    glon = np.arange(0, 360, 0.5)
    GLON, GLAT = np.meshgrid((glon + 180) % 360 - 180, glat)
    hours = g.shape[0]
    u = np.zeros((hours, glat.size, glon.size), np.float32)
    v = np.zeros_like(u)
    for t in range(hours):
        for k, dst in ((0, u), (1, v)):
            vals = g[t, 2 * level + k].astype(np.float32)[pick]
            f = LinearNDInterpolator(tri, vals)(GLON, GLAT)
            hole = ~np.isfinite(f)
            if hole.any():  # the poles sit outside the hull
                f[hole] = NearestNDInterpolator(np.column_stack([plon, plat]), vals)(GLON[hole], GLAT[hole])
            dst[t] = f
        print(f"hour {t}: max speed {np.hypot(u[t], v[t]).max():.1f} m/s", flush=True)
    np.savez_compressed(out, u=u.astype(np.float16), v=v.astype(np.float16), lat=glat, lon=glon, hours=z["hours"], hpa=LEVELS_HPA.get(level, -1))
    print(f"wrote {out}")


# ---------------------------------------------------------------------------------------------- the wind


class Wind:
    """u, v on a 0.5 deg grid over 13 hours, read bilinearly in space and linearly in time"""

    def __init__(self, path):
        z = np.load(path)
        from scipy.ndimage import gaussian_filter

        # a light smoothing hides the row pattern the thinned HEALPix sampling leaves in the regrid
        soft = lambda a: gaussian_filter(a.astype(np.float32), (0, 1.0, 1.0), mode=("nearest", "nearest", "wrap"))
        self.u, self.v = soft(z["u"]), soft(z["v"])
        self.speed = np.hypot(self.u, self.v)
        self.hours = self.u.shape[0]
        self.nlat, self.nlon = self.u.shape[1:]

    def hour(self, frame):
        """The 13 hours forward and back on a cosine, so the loop closes"""
        return 0.5 * (self.hours - 1) * (1 - np.cos(2 * np.pi * (frame % FRAMES) / FRAMES))

    def _index(self, lat, lon):
        i = np.clip((lat + 90) / 0.5, 0, self.nlat - 1.000001)
        j = (lon % 360) / 0.5
        i0, j0 = np.floor(i).astype(np.int64), np.floor(j).astype(np.int64) % self.nlon
        return i0, j0, (j0 + 1) % self.nlon, (i - i0).astype(np.float32), (j - np.floor(j)).astype(np.float32)

    def sample(self, fields, frame, lat, lon):
        h = self.hour(frame)
        t0 = int(min(np.floor(h), self.hours - 2))
        ft = np.float32(h - t0)
        i0, j0, j1, fi, fj = self._index(lat, lon)
        out = []
        for f in fields:
            acc = 0
            for t, wt in ((t0, 1 - ft), (t0 + 1, ft)):
                a = f[t]
                acc = acc + wt * (
                    (1 - fi) * ((1 - fj) * a[i0, j0] + fj * a[i0, j1]) + fi * ((1 - fj) * a[i0 + 1, j0] + fj * a[i0 + 1, j1])
                )
            out.append(acc)
        return out

    def step(self, p, frame, hours):
        """Each point carried `hours` along the great circle its wind points down"""
        lat, lon = latlon(p)
        u, v = self.sample((self.u, self.v), frame, lat, lon)
        la, lo = np.radians(lat), np.radians(lon)
        east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
        north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
        vel = u[:, None] * east + v[:, None] * north
        pace = np.linalg.norm(vel, axis=-1, keepdims=True)
        ang = pace * hours * 3.6 / EARTH_KM
        d = np.divide(vel, pace, out=np.zeros_like(vel), where=pace > 0)
        q = p * np.cos(ang) + d * np.sin(ang)
        return q / np.linalg.norm(q, axis=-1, keepdims=True)


def even_sphere(rng, n):
    return unit(np.degrees(np.arcsin(rng.uniform(-1, 1, n))), rng.uniform(-180, 180, n))


# ---------------------------------------------------------------------------------------------- the camera


class Camera:
    def __init__(self):
        self.S = SIZE * SS
        self.R = RADIUS * self.S
        c = (np.arange(self.S) + 0.5 - self.S / 2) / self.R
        X, Y = np.meshgrid(c, -c)
        rho2 = X**2 + Y**2
        self.inside = rho2 < 1
        Z = np.sqrt(np.clip(1 - rho2, 0, 1))
        self.rho = np.sqrt(rho2)
        self.z = Z
        # with the camera at longitude 0, a pixel's latitude is fixed and its longitude shifts with the turn
        e, n, f = self.basis(0.0)
        p = X[..., None] * e + Y[..., None] * n + Z[..., None] * f
        self.lat, self.lon0 = latlon(p)

    def lon(self, frame):
        return LON0 - 360.0 * (frame % FRAMES) / FRAMES  # the earth's own sense of turning: land moves left to right

    @staticmethod
    def basis(lon_c):
        la, lo = np.radians(TILT), np.radians(lon_c)
        e = np.array([-np.sin(lo), np.cos(lo), 0.0])
        n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
        f = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        return e, n, f

    def project(self, p, frame):
        """Unit vectors to supersampled pixels, and their depth toward the camera (> 0 is the near side)"""
        e, n, f = self.basis(self.lon(frame))
        return self.S / 2 + self.R * (p @ e), self.S / 2 - self.R * (p @ n), p @ f


# ---------------------------------------------------------------------------------------------- the drawing


def splat(buf, x, y, w):
    S = buf.shape[0]
    ok = np.isfinite(x) & (x >= 0) & (x < S - 1) & (y >= 0) & (y < S - 1) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    flat = buf.reshape(-1)
    for dx, dy, s in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        flat += np.bincount((y0 + dy) * S + x0 + dx, weights=s * w, minlength=flat.size).astype(np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def coastlines():
    """Natural Earth 50 m coastlines as segments of unit vectors, (m, 2, 3)"""
    import cartopy.io.shapereader as shp

    segs = []
    for rec in shp.Reader(shp.natural_earth("50m", "physical", "coastline")).geometries():
        for line in getattr(rec, "geoms", [rec]):
            xy = np.asarray(line.coords)
            if len(xy) > 1:
                p = unit(xy[:, 1], xy[:, 0])
                segs.append(np.stack([p[:-1], p[1:]], 1))
    return np.concatenate(segs)


def land_raster(width=2880):
    """Land as an equirectangular mask, lat descending from 90, lon from -180"""
    import cartopy.io.shapereader as shp
    from PIL import Image, ImageDraw

    h = width // 2
    img = Image.new("L", (width, h), 0)
    d = ImageDraw.Draw(img)
    for geom in shp.Reader(shp.natural_earth("50m", "physical", "land")).geometries():
        for poly in getattr(geom, "geoms", [geom]):
            xy = np.asarray(poly.exterior.coords)
            d.polygon([((lo + 180) / 360 * width, (90 - la) / 180 * h) for lo, la in xy], fill=255)
            for ring in poly.interiors:
                xy = np.asarray(ring.coords)
                d.polygon([((lo + 180) / 360 * width, (90 - la) / 180 * h) for lo, la in xy], fill=0)
    return np.asarray(img, np.float32) / 255


class Film:
    def __init__(self, wind_path):
        from scipy.ndimage import gaussian_filter

        self.blur = gaussian_filter
        self.wind = Wind(wind_path)
        self.cam = Camera()
        self.lut = lut()
        self.coast = coastlines()
        land = land_raster()
        self.land_h, self.land_w = land.shape
        self.land = land
        self.background = self._space()
        self._particles()
        self._balloons()

    # -- the still parts
    def _space(self):
        S, cam = self.cam.S, self.cam
        img = np.broadcast_to(SPACE, (S, S, 3)).copy()
        rng = np.random.default_rng(3)
        stars = np.zeros((S, S), np.float32)
        n = 420
        splat(stars, rng.uniform(0, S - 1, n), rng.uniform(0, S - 1, n), rng.uniform(0.15, 1.0, n) ** 2 * 1.6)
        stars = self.blur(stars, 0.7)
        img += np.clip(stars, 0, 1)[..., None] * np.array([210, 220, 255], np.float32)
        out = ~cam.inside
        glow = np.where(out, np.exp(-(cam.rho - 1) / 0.035), 0)
        img = img * (1 - 0.55 * glow[..., None]) + ATMOSPHERE * 0.55 * glow[..., None]
        return img

    def _particles(self):
        rng = np.random.default_rng(11)
        cycles = FRAMES // LIFE
        self.sow = np.stack([even_sphere(rng, PARTICLES) for _ in range(cycles)])  # (cycles, n, 3)
        self.phase = rng.integers(0, LIFE, PARTICLES)
        # the run before frame 0 that fills every trail
        self.hist = np.zeros((TRAIL, PARTICLES, 3))
        self.age = np.full(PARTICLES, -1)
        self.p = None
        self.frame = -LIFE - 1
        self._advance()

    def _advance(self):
        """One frame on: every particle carried, the ones whose time is up sown afresh"""
        f = self.frame + 1
        reborn = (f + self.phase) % LIFE == 0
        if self.p is None:
            self.p = even_sphere(np.random.default_rng(0), PARTICLES)  # placeholders, never drawn before their first sowing
        else:
            self.p = self.wind.step(self.p, f - 1, STEP_HOURS)
        k = ((f + self.phase) // LIFE) % (FRAMES // LIFE)
        self.p[reborn] = self.sow[k[reborn], np.flatnonzero(reborn)]
        self.age = np.where(reborn, 0, np.where(self.age >= 0, self.age + 1, -1))
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = self.p
        self.frame = f

    def seek(self, frame):
        if frame < self.frame:
            self._particles()
        while self.frame < frame:
            self._advance()

    def _balloons(self):
        """Each balloon's whole life, centred in view at mid-life, carried forward and back from there"""
        rng = np.random.default_rng(5)
        lats = np.array([-56.0, 14.0, -42.0, -63.0, 0.0, -50.0, 28.0, -35.0])[:BALLOONS]
        self.tracks = []
        self.births = [j * FRAMES // BALLOONS for j in range(BALLOONS)]
        for j, b in enumerate(self.births):
            mid = b + BALLOON_LIFE // 2
            lon = self.cam.lon(mid) + rng.uniform(-25, 25)
            start = unit(np.array([lats[j]]), np.array([lon]))
            track = np.zeros((BALLOON_LIFE, 3))
            m = BALLOON_LIFE // 2
            track[m] = start[0]
            p = start
            for a in range(m + 1, BALLOON_LIFE):
                p = self.wind.step(p, b + a - 1, STEP_HOURS)
                track[a] = p[0]
            p = start
            for a in range(m - 1, -1, -1):
                p = self.wind.step(p, b + a, -STEP_HOURS)
                track[a] = p[0]
            self.tracks.append(track)

    # -- one frame
    def globe(self, frame):
        cam, w = self.cam, self.wind
        lon = cam.lon0 + cam.lon(frame)
        (spd,) = w.sample((w.speed,), frame, cam.lat, lon)
        x = np.clip(spd / VMAX, 0, 1) ** GAMMA
        col = self.lut[(x * 255).astype(np.int64)]
        li = np.clip(((90 - cam.lat) / 180 * self.land_h).astype(np.int64), 0, self.land_h - 1)
        lj = (((lon + 180) % 360) / 360 * self.land_w).astype(np.int64) % self.land_w
        land = self.land[li, lj][..., None]
        col = col * (1 - 0.15 * land) + LAND * 0.15 * land
        shade = (0.62 + 0.38 * cam.z**0.5)[..., None]
        col = col * shade
        rim = ((1 - cam.z) ** 4)[..., None]
        col = col * (1 - 0.35 * rim) + ATMOSPHERE * 0.35 * rim
        edge = smoothstep(1.0, 1.0 - 1.5 / cam.R, cam.rho)[..., None]  # an antialiased limb
        return self.background * (1 - edge) + col * edge

    def coastline_mask(self, frame):
        from PIL import Image, ImageDraw

        S = self.cam.S
        a = self.coast.reshape(-1, 3)
        x, y, z = self.cam.project(a, frame)
        x, y, z = x.reshape(-1, 2), y.reshape(-1, 2), z.reshape(-1, 2)
        keep = (z > 0.03).all(1)
        img = Image.new("L", (S, S), 0)
        d = ImageDraw.Draw(img)
        for (x0, x1), (y0, y1) in zip(x[keep], y[keep]):
            d.line((x0, y0, x1, y1), fill=255, width=SS)
        return np.asarray(img, np.float32) / 255 * smoothstep(0.0, 0.25, self.cam.z)

    def streaks(self, frame):
        S, cam = self.cam.S, self.cam
        buf = np.zeros((S, S), np.float32)
        lat, lon = latlon(self.hist[0])
        (spd,) = self.wind.sample((self.wind.speed,), frame, lat, lon)
        bright = 0.04 + 1.1 * np.clip(spd / VMAX, 0, 1) ** 1.3  # jets glow, calms barely show
        life = smoothstep(0, 4, self.age) * smoothstep(LIFE - 1, LIFE - 7, self.age)
        for k in range(TRAIL - 1):
            ok = self.age >= k + 1
            if not ok.any():
                continue
            fade = (1 - k / (TRAIL - 1)) ** 1.4
            xa, ya, za = cam.project(self.hist[k][ok], frame)
            xb, yb, zb = cam.project(self.hist[k + 1][ok], frame)
            wt = (bright * life)[ok] * fade * smoothstep(0.0, 0.2, np.minimum(za, zb))
            jump = np.hypot(xa - xb, ya - yb) > 0.2 * S
            wt = np.where(jump, 0, wt)
            for s in np.linspace(0, 1, 5, endpoint=False):
                splat(buf, xa + (xb - xa) * s, ya + (yb - ya) * s, wt * 1.8)
        return buf

    def balloons(self, frame):
        """(track buffer, list of (x, y, alpha)) at this frame"""
        S, cam = self.cam.S, self.cam
        buf = np.zeros((S, S), np.float32)
        dots = []
        for b, track in zip(self.births, self.tracks):
            age = (frame - b) % FRAMES
            if age >= BALLOON_LIFE:
                continue
            fade = smoothstep(0, 18, age) * smoothstep(BALLOON_LIFE - 1, BALLOON_LIFE - 18, age)
            lo = max(0, age - BALLOON_TRAIL)
            pts = track[lo : age + 1]
            x, y, z = cam.project(pts, frame)
            if len(pts) > 1:
                t = np.linspace(0, 1, len(pts) - 1)  # 0 oldest, 1 newest
                vis = smoothstep(0.0, 0.2, np.minimum(z[:-1], z[1:]))
                for s in np.linspace(0, 1, 6, endpoint=False):
                    splat(buf, x[:-1] + (x[1:] - x[:-1]) * s, y[:-1] + (y[1:] - y[:-1]) * s, (t**1.2) * vis * fade * 9.0)
            dots.append((x[-1], y[-1], fade * smoothstep(0.0, 0.2, z[-1])))
        return buf, dots

    def render(self, frame):
        self.seek(frame)
        S = self.cam.S
        img = self.globe(frame)
        coast = self.coastline_mask(frame)[..., None] * 0.42
        img = img * (1 - coast) + COAST * coast
        streak = self.streaks(frame)
        streak = self.blur(streak, 1.0 * SS) * 1.0 + self.blur(streak, 3.0 * SS) * 0.35
        a = (1 - np.exp(-1.6 * streak))[..., None]
        img = img * (1 - a) + STREAK * a
        track, dots = self.balloons(frame)
        track = self.blur(track, 1.2 * SS)
        a = (1 - np.exp(-1.4 * track))[..., None]
        img = img * (1 - a) + TRACK * a
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        for x, y, alpha in dots:
            if alpha <= 0.01:
                continue
            r = 15 * SS
            ys, xs = slice(max(0, int(y) - 3 * r), int(y) + 3 * r), slice(max(0, int(x) - 3 * r), int(x) + 3 * r)
            d = np.hypot(xx[ys, xs] - x, yy[ys, xs] - y)
            halo = np.exp(-((d / r) ** 2)) * 0.45 * alpha
            core = smoothstep(5.6 * SS, 4.4 * SS, d) * alpha
            sub = img[ys, xs]
            sub[:] = sub * (1 - halo[..., None]) + BALLOON * halo[..., None]
            ring = smoothstep(7.2 * SS, 6.0 * SS, d) * alpha * 0.8  # a dark ring so the dot reads on the bright jets
            sub[:] = sub * (1 - ring[..., None]) + np.array([10, 14, 24], np.float32) * ring[..., None]
            sub[:] = sub * (1 - core[..., None]) + BALLOON * core[..., None]
        img = img.reshape(SIZE, SS, SIZE, SS, 3).mean(axis=(1, 3))
        return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- the command


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--level", type=int, default=0)
    s = sub.add_parser("still")
    s.add_argument("wind")
    s.add_argument("out")
    s.add_argument("--frame", type=int, default=0)
    r = sub.add_parser("render")
    r.add_argument("wind")
    r.add_argument("video")
    r.add_argument("poster")
    r.add_argument("--crf", type=int, default=27)
    a = ap.parse_args()
    if a.cmd == "prepare":
        return prepare(a.src, a.out, a.level)
    from PIL import Image

    film = Film(a.wind)
    if a.cmd == "still":
        Image.fromarray(film.render(a.frame)).save(a.out)
        return
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryslow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-movflags", "+faststart", "-an", a.video],
        stdin=subprocess.PIPE,
    )
    for f in range(FRAMES):
        frame = film.render(f)
        if f == 0:
            Image.fromarray(frame).save(a.poster, quality=86, optimize=True, progressive=True)
        enc.stdin.write(frame.tobytes())
        if f % 24 == 0:
            print(f"frame {f}/{FRAMES}", flush=True)
    enc.stdin.close()
    sys.exit(enc.wait())


if __name__ == "__main__":
    main()
