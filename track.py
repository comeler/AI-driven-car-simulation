"""
Track: a closed circuit represented by two boundary polylines (inner wall and
outer wall) plus a series of checkpoints for measuring progress around the
loop. The Track class only cares about these two polylines + checkpoints -
how they were generated doesn't matter, so you can hand-edit inner/outer
later to build arbitrary (non-oval) track shapes.
"""

import math


def _flip_direction(centerline):
    """Reverse travel direction around a closed loop while keeping the same
    starting point, so the car still starts at the same spot, facing the
    (now opposite) direction of travel."""
    return [centerline[0]] + list(reversed(centerline[1:]))


def generate_oval_centerline(cx, cy, rx, ry, num_points=120, clockwise=True):
    """Convenience: generate a simple oval centerline to get a track running
    quickly. Swap this out later for a hand-designed list of (x, y) points."""
    points = []
    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        points.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
    return points if clockwise else _flip_direction(points)


def generate_wavy_centerline(cx, cy, rx, ry, wobble=22, frequency=5, num_points=220, clockwise=True):
    """Oval whose radius wobbles sinusoidally - curvature keeps changing
    continuously around the whole lap, so a fixed turn rate can never match
    the track for more than an instant. Harder than a plain oval.
    Caution: wobble too large relative to track width makes the offset
    boundaries fold over themselves (self-intersect) at the tightest points -
    keep wobble well under the track width, or widen the track."""
    points = []
    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        scale = 1 + (wobble / max(rx, ry)) * math.sin(t * frequency)
        points.append((cx + rx * math.cos(t) * scale, cy + ry * math.sin(t) * scale))
    return points if clockwise else _flip_direction(points)


def generate_rounded_rect_centerline(cx, cy, width, height, corner_radius, num_points=160, clockwise=True):
    """Rounded rectangle: 4 straight edges connected by 4 constant-radius
    corners. Harder than the oval because curvature is not smooth - it jumps
    abruptly between 'straight, no turning' and 'sharp constant turn',
    instead of changing gradually."""
    hw = width / 2 - corner_radius
    hh = height / 2 - corner_radius
    r = corner_radius

    # 4 corner arc centers, going clockwise from top-right
    corner_specs = [
        ((cx + hw, cy - hh), -90, 0),
        ((cx + hw, cy + hh), 0, 90),
        ((cx - hw, cy + hh), 90, 180),
        ((cx - hw, cy - hh), 180, 270),
    ]
    arc_samples = max(6, num_points // 20)
    raw_points = []
    for (ccx, ccy), a0, a1 in corner_specs:
        for i in range(arc_samples):
            t = i / arc_samples
            angle = math.radians(a0 + t * (a1 - a0))
            raw_points.append((ccx + r * math.cos(angle), ccy + r * math.sin(angle)))
    # The straight edges are implicit: _resample_uniform connects consecutive
    # raw points with straight lines, and the gap between the last sample of
    # one corner's arc and the first sample of the next corner IS a straight
    # edge, so we don't need to sample the straights explicitly.
    points = _resample_uniform(raw_points, num_points)
    return points if clockwise else _flip_direction(points)


def generate_pinched_loop_centerline(cx, cy, rx, ry, num_pinches=3, pinch_strength=0.25,
                                      num_points=240, clockwise=True):
    """A 'gear-like' loop with several tight pinches around the lap - the
    hardest of these shapes, since curvature alternates repeatedly between
    wide open sweeps and sharp narrow turns. Keep pinch_strength modest and
    track width generous (>= 90-100), or the inner/outer boundaries can
    self-intersect at the tightest points (see the Option B caveat about
    sharp corners)."""
    points = []
    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        scale = 1 - pinch_strength * (0.5 + 0.5 * math.cos(t * num_pinches))
        points.append((cx + rx * math.cos(t) * scale, cy + ry * math.sin(t) * scale))
    return points if clockwise else _flip_direction(points)


def _resample_uniform(points, num_points):
    """Resample a closed polyline to num_points, evenly spaced by arc length.
    Needed because a plain parametric oval (equal steps in angle t) packs
    points closer together near the ends of the major axis and further apart
    near the ends of the minor axis - left as-is, that uneven spacing would
    make checkpoints unevenly spaced too, which is exactly what caused a bug
    below (see ProgressTracker)."""
    n = len(points)
    cumulative = [0.0]
    for i in range(1, n + 1):
        p0, p1 = points[(i - 1) % n], points[i % n]
        cumulative.append(cumulative[-1] + math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    total_length = cumulative[-1]
    step = total_length / num_points

    resampled = []
    seg = 0
    for k in range(num_points):
        target = k * step
        while cumulative[seg + 1] < target:
            seg += 1
        seg_start = cumulative[seg]
        seg_len = (cumulative[seg + 1] - seg_start) or 1e-9
        t = (target - seg_start) / seg_len
        p0, p1 = points[seg % n], points[(seg + 1) % n]
        resampled.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))
    return resampled


def _offset_centerline(centerline, width):
    """Offset a closed centerline left/right by width/2 to build two boundary
    polylines. Only used to bootstrap a track from a centerline."""
    n = len(centerline)
    inner, outer = [], []
    for i in range(n):
        p_prev = centerline[i - 1]
        p_next = centerline[(i + 1) % n]
        dx, dy = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length  # normal (perpendicular) direction
        cx, cy = centerline[i]
        # Note: for this parametrization, +normal points toward the track
        # center, so it defines the inner wall (and -normal the outer wall).
        inner.append((cx + nx * width / 2, cy + ny * width / 2))
        outer.append((cx - nx * width / 2, cy - ny * width / 2))
    return inner, outer


def _point_segment_distance(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    denom = abx ** 2 + aby ** 2 or 1.0
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    closest = (ax + t * abx, ay + t * aby)
    return math.hypot(px - closest[0], py - closest[1])


def _ray_segment_intersection(ox, oy, dx, dy, ax, ay, bx, by):
    """Distance along the ray (ox,oy) + (dx,dy)*t to its intersection with
    segment AB, or None if they don't intersect in front of the ray."""
    sx, sy = bx - ax, by - ay
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None
    t = ((ax - ox) * sy - (ay - oy) * sx) / denom
    u = ((ax - ox) * dy - (ay - oy) * dx) / denom
    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def _point_in_polygon(x, y, polygon):
    """Even-odd ray casting test: is (x, y) inside the closed polygon?"""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


class Track:
    def __init__(self, inner, outer, checkpoints):
        self.inner = inner              # list of (x, y) - inner wall
        self.outer = outer              # list of (x, y) - outer wall
        self.checkpoints = checkpoints  # list of (x, y) - progress markers, in order

    @classmethod
    def from_centerline(cls, centerline, width=120, num_checkpoints=24):
        """Build a Track by offsetting a centerline. Convenience constructor -
        the resulting Track only stores/uses inner, outer and checkpoints, so
        you could equally construct one by hand-picking boundary points."""
        centerline = _resample_uniform(centerline, len(centerline))
        inner, outer = _offset_centerline(centerline, width)
        step = max(1, len(centerline) // num_checkpoints)
        checkpoints = [centerline[i] for i in range(0, len(centerline), step)]
        return cls(inner, outer, checkpoints)

    def is_on_track(self, x, y):
        """True if (x, y) is in the drivable corridor: inside the outer wall
        and outside the inner wall."""
        return _point_in_polygon(x, y, self.outer) and not _point_in_polygon(x, y, self.inner)

    def distance_to_walls(self, x, y):
        """Shortest distance from (x, y) to either boundary polyline."""
        best = float("inf")
        for boundary in (self.inner, self.outer):
            n = len(boundary)
            for i in range(n):
                a, b = boundary[i], boundary[(i + 1) % n]
                best = min(best, _point_segment_distance(x, y, a[0], a[1], b[0], b[1]))
        return best

    def cast_sensor(self, x, y, angle, max_range=300):
        """Cast one ray from (x, y) at `angle`, return distance to the
        nearest wall hit, or max_range if nothing is hit within range."""
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_range
        for boundary in (self.inner, self.outer):
            n = len(boundary)
            for i in range(n):
                a, b = boundary[i], boundary[(i + 1) % n]
                t = _ray_segment_intersection(x, y, dx, dy, a[0], a[1], b[0], b[1])
                if t is not None and t < best:
                    best = t
        return best

    def get_sensor_readings(self, x, y, heading, sensor_angles_deg=(-90, -45, 0, 45, 90), max_range=300):
        """Read several sensors at once, at angles (degrees) relative to the
        car's heading. Returns a list of distances, same order as sensor_angles_deg."""
        return [
            self.cast_sensor(x, y, heading + math.radians(a), max_range)
            for a in sensor_angles_deg
        ]


class ProgressTracker:
    """Tracks how far a car has progressed around a Track's checkpoints.
    Kept separate from Track itself since progress is per-car, per-run state,
    not a property of the track layout."""

    def __init__(self, track, checkpoint_radius=40):
        self.track = track
        # The car starts AT checkpoint 0 (the start line), so it's already
        # "passed" it - begin targeting checkpoint 1 instead. Otherwise the
        # car gets a free checkpoint bonus at reset() without moving at all.
        self.next_checkpoint = 1 % len(track.checkpoints)
        self.laps_completed = 0
        self.checkpoint_radius = checkpoint_radius

    def update(self, x, y):
        target = self.track.checkpoints[self.next_checkpoint]
        dist = math.hypot(x - target[0], y - target[1])
        if dist < self.checkpoint_radius:
            self.next_checkpoint += 1
            if self.next_checkpoint >= len(self.track.checkpoints):
                self.next_checkpoint = 0
                self.laps_completed += 1

    @property
    def progress_fraction(self):
        """Progress within the current lap, as a 0-1 fraction."""
        return self.next_checkpoint / len(self.track.checkpoints)


def build_track_from_config(cfg):
    """Build a Track using the shape and parameters specified in config.py -
    this is the one place that dispatches TRACK_SHAPE to the right generator,
    so switching shapes is just editing config.py, nothing else."""
    shape = cfg.TRACK_SHAPE

    if shape == "oval":
        centerline = generate_oval_centerline(
            cfg.TRACK_CENTER_X, cfg.TRACK_CENTER_Y, cfg.TRACK_RX, cfg.TRACK_RY,
            num_points=cfg.TRACK_NUM_POINTS, clockwise=cfg.TRACK_CLOCKWISE,
        )
    elif shape == "wavy":
        centerline = generate_wavy_centerline(
            cfg.TRACK_CENTER_X, cfg.TRACK_CENTER_Y, cfg.TRACK_RX, cfg.TRACK_RY,
            wobble=cfg.TRACK_WOBBLE, frequency=cfg.TRACK_WOBBLE_FREQUENCY,
            num_points=cfg.TRACK_NUM_POINTS, clockwise=cfg.TRACK_CLOCKWISE,
        )
    elif shape == "rounded_rect":
        centerline = generate_rounded_rect_centerline(
            cfg.TRACK_CENTER_X, cfg.TRACK_CENTER_Y, cfg.TRACK_RECT_WIDTH, cfg.TRACK_RECT_HEIGHT,
            corner_radius=cfg.TRACK_CORNER_RADIUS,
            num_points=cfg.TRACK_NUM_POINTS, clockwise=cfg.TRACK_CLOCKWISE,
        )
    elif shape == "pinched":
        centerline = generate_pinched_loop_centerline(
            cfg.TRACK_CENTER_X, cfg.TRACK_CENTER_Y, cfg.TRACK_RX, cfg.TRACK_RY,
            num_pinches=cfg.TRACK_NUM_PINCHES, pinch_strength=cfg.TRACK_PINCH_STRENGTH,
            num_points=cfg.TRACK_NUM_POINTS, clockwise=cfg.TRACK_CLOCKWISE,
        )
    else:
        raise ValueError(
            f"Unknown TRACK_SHAPE '{shape}' in config.py - "
            "use 'oval', 'wavy', 'rounded_rect' or 'pinched'."
        )

    return Track.from_centerline(centerline, width=cfg.TRACK_CORRIDOR_WIDTH,
                                  num_checkpoints=cfg.TRACK_NUM_CHECKPOINTS)
