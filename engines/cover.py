"""Engine: cover.

After pysal/spopt (BSD-3-Clause) for location-allocation, and
PedestrianDynamics/jupedsim (LGPL-3.0-or-later) for checking the answer by
walking a crowd through it.

Every other engine in this project answers "does it fit". For half a programme
that is the wrong question. Nobody cares how tightly a toilet block, a bin, a
first-aid post, a mist fan or a shade sail packs; what matters is that no
attendee is further than some walk from one. That is a covering problem, and
it has had an exact formulation since Toregas 1971 and Church & ReVelle 1974,
both of which spopt ships:

    LSCP   the fewest facilities such that every demand point is within R
    MCLP   given exactly p facilities, the most demand covered within R

So the engine works in three layers:

    demand      the free ground sampled on a 10 m lattice, each point weighted
                by the open ground within 20 m of it, so the middle of a plaza
                outvotes a sliver behind a container
    location    LSCP to learn how many units this ground actually needs, MCLP
                to spend the number the brief bought
    geometry    the chosen sites realised as legal placements, because a
                covering model knows nothing about a 9.6 m toilet block
                standing on top of another one

Everything not coverage-shaped — the stage, the seating, back of house — goes
through a plain greedy, biggest first: the anchor with the most weighted
standing ground within 60 m, pulled towards whatever it wants to be near and
pushed off its own kind. That half of the programme gets no cleverness, and
the notes name every group that went through it, rather than let a good
coverage number stand in for a good plan.

`egress` is separate from `solve`, and it is the other half of the argument.
It takes a finished layout, subtracts every footprint from the ground, and
walks a crowd out with jupedsim's collision-free speed model. No packing check
can tell you that the last rank of food trucks closed the only way out of the
north plaza; reachability can, and this one is measured against the bare site
so the answer is what the LAYOUT closed and not what the drawing already had.
Tested against a deliberately fatal plan — a wall of barriers across the site
with the only gate behind it — it reports 6 827 m2 and 4 875 people cut off
where the same site with no layout reports none.
"""

from __future__ import annotations

import math
import re
import time

from core.placement import Ground, Placement, anchors

NAME = "cover"
DOC = ("Location-allocation: LSCP and MCLP over weighted demand points, for "
       "the items whose requirement is reach rather than fit — toilets, bins, "
       "first aid, misting, shade — and a greedy on reachable standing ground "
       "for the rest. Reports the service radius it achieved and the "
       "worst-served point on the site, names every group it could not model, "
       "and can check a finished layout by simulating the crowd leaving it. "
       "Deterministic: the same site and programme give the same plan.")

# What "coverage-shaped" means, and how far a person may reasonably be asked
# to walk to one, on an outdoor festival site in a hot climate. These are
# working defaults, not regulation: the engine reports the radius it actually
# achieved so the number can be argued with rather than believed. Anything
# whose kind is absent from this table is packed, not covered.
SERVICE_RADIUS = {
    "sanitary": 75.0,     # a toilet walk people will accept before queueing elsewhere
    "waste": 40.0,        # rubbish is carried in the hand, and not far
    "medical": 120.0,     # first aid is a response time, not a queue
    "water": 60.0,
    "cooling": 45.0,      # a mist fan cools the people standing in it, nobody else
    "shade": 40.0,        # in Jeddah in summer shade is infrastructure, not decoration
    "exit": 150.0,
}

DEMAND_CELL = 2.0        # m, the raster the openness weight is measured on
DEMAND_SPACING = 10.0    # m, the lattice demand points sit on
OPEN_RADIUS = 20.0       # m, the disc openness is measured over
CROWD_FLOOR = 200.0      # m2 within OPEN_RADIUS before a point counts as standing room
MAX_CANDIDATES = 140     # facility sites offered to the LP
MAX_ANCHORS = 4000       # legal positions kept per group
NEAR_RANGE = 60.0        # m, what `near` means — the same distance `score` uses


# ── where the crowd is ───────────────────────────────────────────────────

class Crowd:
    """The free ground as demand: where people can stand, and how much room.

    A covering model treats every demand point alike, so the weights are the
    whole argument. Unweighted, this site's 149 disjoint free parts would let
    101 slivers under 5 m2 outvote the plazas, and the toilets would end up
    ringing the rubbish. The weight is the area of free ground within 20 m of
    the point, which is a direct measure of how many people can be there.
    """

    def __init__(self, free, region, cell=DEMAND_CELL, spacing=DEMAND_SPACING,
                 radius=OPEN_RADIUS):
        import numpy as np
        import shapely
        from scipy import ndimage

        x0, y0, x1, y1 = region
        self.cell = float(cell)
        self.x0, self.y0 = float(x0), float(y0)
        self.nx = max(1, int((x1 - x0) / self.cell))
        self.ny = max(1, int((y1 - y0) / self.cell))
        gx = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        gy = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        XX, YY = np.meshgrid(gx, gy, indexing="ij")
        # One vectorised containment call for the whole raster; the same trick
        # `Ground` uses, and for the same reason.
        inside = shapely.contains_xy(
            free, XX.ravel(), YY.ravel()).reshape(self.nx, self.ny)

        # A disc kernel, not a box: a box reads a 40 m corridor as being as
        # open as a plaza, and telling those two apart is the entire job of
        # this weight.
        r = max(1, int(round(radius / self.cell)))
        ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
        disc = ((ii ** 2 + jj ** 2) <= r * r).astype(np.float32)
        self.open_m2 = ndimage.convolve(inside.astype(np.float32), disc,
                                        mode="constant") * self.cell ** 2

        k = max(1, int(round(spacing / self.cell)))
        ci, cj = np.nonzero(inside)
        keep = (ci % k == 0) & (cj % k == 0)
        ci, cj = ci[keep], cj[keep]
        self.pts = np.column_stack([self.x0 + (ci + 0.5) * self.cell,
                                    self.y0 + (cj + 0.5) * self.cell])
        self.weight = self.open_m2[ci, cj].astype(float)
        # Standing room, as opposed to ground a person could technically stand
        # on. LSCP has no weights — it insists every point it is given is
        # covered — so handing it the slivers would make the model demand a
        # toilet behind every container, or come back infeasible.
        self.standing = self.weight >= CROWD_FLOOR

    def served(self, xs, ys, radius=NEAR_RANGE):
        """Weighted standing ground within `radius` of each (x, y).

        The measure the greedy half of this engine ranks positions by, and the
        first thing that had to change after looking at a render: ranking on
        raw openness put the main stage in a 35 m pocket at the east end of
        the site, which is genuinely the most open ground per square metre and
        is cut off from the 240 m band where the crowd actually is. Openness
        is a local property; being in the middle of the event is not."""
        import numpy as np
        from scipy.spatial.distance import cdist
        d = cdist(np.column_stack([np.asarray(xs, float), np.asarray(ys, float)]),
                  self.pts[self.standing])
        return (d <= radius) @ self.weight[self.standing]


# ── candidate sites ──────────────────────────────────────────────────────

def _ground_for(free, occupied, index, item):
    """The ground this item may stand on, with what is already there removed.

    An overhead object — a sail, a canopy — is exempt from every collision
    check in `validate`, so it may span anything on the ground. It is not
    exempt from taste: six sails stacked in the same corner is not a plan. So
    the subtraction rule is one line — remove what shares this item's overhead
    flag — which keeps sails off each other and off nothing else.
    """
    from shapely.ops import unary_union
    taken = []
    for p in occupied:
        other = index.get(p.key)
        if other is None or other.overhead != item.overhead:
            continue
        # The candidate already carries item.clearance/2 as an `anchors`
        # margin, so the ground has to give back the REST of the larger of the
        # two clearances. Halving the larger instead leaves the pair short
        # whenever the thing already standing wants more room than the thing
        # being placed: a 6 m keep-out met by a 0.5 m one came out 3.25 m
        # apart, and `validate` calls that a clearance failure.
        taken.append(p.footprint(
            other, pad=max(item.clearance, other.clearance) - item.clearance / 2))
    if not taken:
        return free
    blocked = unary_union(taken)
    return free if blocked.is_empty else free.difference(blocked)


def _separation(a, b):
    """Metres that must separate two items, EDGE to edge, in either direction.

    Symmetric on purpose. A rule declared on the toilet still binds the food
    truck, whichever of the two this engine happens to be placing now."""
    d = 0.0
    if a.min_far > 0 and (b.key in a.far or b.kind in a.far):
        d = a.min_far
    if b.min_far > 0 and (a.key in b.far or a.kind in b.far):
        d = max(d, b.min_far)
    return d


def _sizes(cands, item):
    """Per-candidate (w, h), since a candidate carries the rotation it fits at.

    Multiples of 90 are the whole table in practice and get the vectorised
    path. An oblique rotation falls back to `Item.size`, which returns the
    bounding extent — so the collision test built on it is conservative rather
    than wrong, which is the direction to be wrong in."""
    import numpy as np
    rot = np.array([c[2] for c in cands], dtype=float)
    if np.any(rot % 90):
        wh = np.array([item.size(r) for r in rot], dtype=float)
        return wh[:, 0], wh[:, 1]
    swap = (rot % 180) != 0
    return np.where(swap, item.h, item.w), np.where(swap, item.w, item.h)


def _drop_far(cands, item, occupied, index):
    """Candidates that break a separation rule against something already down.

    Measured with `shapely.distance` between the two footprints, because the
    rule and `validate` both mean edge to edge. Separating centres by 20 m
    puts a 9.6 m toilet 13 m from a 7 m food truck and reports success."""
    import numpy as np
    import shapely
    if not cands:
        return cands
    keep = np.ones(len(cands), dtype=bool)
    w, h = _sizes(cands, item)
    xs = np.array([c[0] for c in cands])
    ys = np.array([c[1] for c in cands])
    boxes = None
    for p in occupied:
        other = index.get(p.key)
        if other is None:
            continue
        d = _separation(item, other)
        if d <= 0:
            continue
        if boxes is None:
            boxes = shapely.box(xs - w / 2, ys - h / 2, xs + w / 2, ys + h / 2)
        keep &= shapely.distance(boxes, p.footprint(other)) >= d
    return [c for c, ok in zip(cands, keep) if ok]


def _thin(cands, k, start=0):
    """Farthest-point thinning down to k candidates.

    The LP cost is quadratic in the number of sites offered and this site
    yields thousands of legal ones. Taking every nth in raster order gives a
    set that is dense along rows and blind between them; maximin sampling
    gives one that is spread, which is what a covering model wants to choose
    from. Deterministic, given the same starting index."""
    import numpy as np
    if len(cands) <= k:
        return list(range(len(cands)))
    P = np.array([(c[0], c[1]) for c in cands], dtype=float)
    keep = [int(start)]
    d = np.hypot(P[:, 0] - P[start, 0], P[:, 1] - P[start, 1])
    for _ in range(k - 1):
        i = int(d.argmax())
        keep.append(i)
        d = np.minimum(d, np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1]))
    return keep


def _clash(cands, item, done):
    """Which candidates collide with, or crowd, what this group already placed.

    Axis-aligned rectangles at 0/90 degrees, so the exact test is arithmetic:
    two boxes are clear when they are apart on either axis. Vectorised because
    the greedy runs it once per item over a few thousand candidates."""
    import numpy as np
    bad = np.zeros(len(cands), dtype=bool)
    if not done:
        return bad
    xs = np.array([c[0] for c in cands])
    ys = np.array([c[1] for c in cands])
    w, h = _sizes(cands, item)
    # A separation rule an item declares against its own kind binds its
    # siblings as well. `_drop_far` only sees what earlier groups put down, so
    # without this four generators told to stand 30 m from power stood 21.6 m
    # apart and `validate` reported it.
    apart = _separation(item, item)
    for (dx, dy, dw, dh, dclr) in done:
        gap = max(item.clearance, dclr, apart)
        bad |= ((np.abs(xs - dx) < (w + dw) / 2 + gap)
                & (np.abs(ys - dy) < (h + dh) / 2 + gap))
    return bad


def _one_size(c, item):
    return item.size(c[2])


def _room(cands, item, dep_cands, dep_item, radius=NEAR_RANGE):
    """For each candidate, how many places the things that must sit beside it
    would still have within `radius` afterwards.

    Without this the greedy puts the stage on the best ground in the middle of
    the one 240 m band, and the nearest position a 30 m grandstand still fits
    is 108 m away — measured, not guessed. A stage that leaves its seating
    nowhere to stand is a worse stage than one 30 m off the sweet spot. It is
    a count of anchors rather than a proof that three grandstands fit; the
    exact question is the packing problem this engine is deliberately not
    solving, and the count is enough to tell a good position from a fatal one."""
    import numpy as np
    if not dep_cands:
        return np.zeros(len(cands))
    keep = _thin(dep_cands, 600)
    dep_cands = [dep_cands[i] for i in keep]
    ax = np.array([c[0] for c in cands], dtype=np.float32)
    ay = np.array([c[1] for c in cands], dtype=np.float32)
    aw, ah = _sizes(cands, item)
    bx = np.array([c[0] for c in dep_cands], dtype=np.float32)
    by = np.array([c[1] for c in dep_cands], dtype=np.float32)
    bw, bh = _sizes(dep_cands, dep_item)
    dx = np.abs(ax[:, None] - bx[None, :])
    dy = np.abs(ay[:, None] - by[None, :])
    gap = max(item.clearance, dep_item.clearance)
    clash = ((dx < (aw[:, None] + bw[None, :]) / 2 + gap)
             & (dy < (ah[:, None] + bh[None, :]) / 2 + gap))
    near = (dx * dx + dy * dy) <= radius * radius
    return (near & ~clash).sum(axis=1).astype(float)


# ── the covering models ──────────────────────────────────────────────────

def _cover_sites(cost, weight, n_new, n_open, radius, budget):
    """Choose n_new sites. Returns (chosen columns, note fragments).

    `cost` is demand x facility; the last `n_open` columns are facilities of
    this kind already standing, handed to the model as predefined so it spends
    the new units on the ground the old ones miss rather than re-deciding a
    layout it cannot change.

    LSCP first, for a fact worth knowing whatever it says: how many units this
    ground needs at this radius. Then MCLP, because the brief bought a number
    and the engine's job is to spend it well, not to argue about it."""
    import numpy as np
    import pulp
    from spopt.locate import LSCP, MCLP

    said = []
    n_fac = cost.shape[1]
    pre = None
    if n_open:
        pre = np.zeros(n_fac, dtype=int)
        pre[n_fac - n_open:] = 1

    # LSCP insists on covering every point it is given, so it only sees the
    # points something can reach. The count of the rest is the honest half of
    # the answer.
    reach = cost.min(axis=1) <= radius
    if reach.any():
        try:
            m = LSCP.from_cost_matrix(cost[reach], radius,
                                      predefined_facilities_arr=pre)
            m = m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=max(2.0, budget / 2)))
            if m.problem.status == 1:
                k = int(round(sum((v.value() or 0) for v in m.fac_vars)))
                said.append(f"LSCP: {k} would cover all reachable standing ground "
                            f"at {radius:g} m")
        except Exception as e:                   # noqa: BLE001 - CBC is a subprocess
            said.append(f"LSCP unavailable ({type(e).__name__})")
    if not reach.all():
        said.append(f"{int((~reach).sum())} of {len(reach)} demand points are "
                    f"beyond {radius:g} m of any legal site")

    p = min(n_new + n_open, n_fac)
    w = np.maximum(weight, 1e-6)
    standing = set(range(n_fac - n_open, n_fac))     # columns already built
    try:
        m = MCLP.from_cost_matrix(cost, w, service_radius=radius, p_facilities=p,
                                  predefined_facilities_arr=pre)
        m = m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=max(3.0, budget)))
        if m.problem.status != 1:
            raise RuntimeError(pulp.LpStatus[m.problem.status])
        chosen = [i for i, v in enumerate(m.fac_vars) if (v.value() or 0) > 0.5]
        chosen = [i for i in chosen if i not in standing]
        said.append(f"MCLP: {m.perc_cov:.0f} % of weighted demand within {radius:g} m")
        if len(chosen) < n_new:
            # MCLP opens exactly p facilities, so a short answer means CBC
            # returned something odd rather than that the ground was full.
            chosen += [i for i in _greedy_sites(cost, w, radius, n_new - len(chosen),
                                                skip=standing | set(chosen))
                       if i not in chosen]
        return chosen[:n_new], said
    except Exception as e:                           # noqa: BLE001
        said.append(f"MCLP fell back to greedy covering ({type(e).__name__}: {e})")
        return _greedy_sites(cost, w, radius, n_new, skip=standing), said


def _greedy_sites(cost, weight, radius, n, skip=()):
    """Classic greedy maximum coverage, as the fallback when CBC will not run.

    Within a factor of 1 - 1/e of optimal, which is a fair answer and a much
    better one than no answer at all."""
    import numpy as np
    covers = cost <= radius
    left = weight.astype(float).copy()
    out = []
    for _ in range(n):
        gain = covers.T @ left
        for i in skip:
            gain[i] = -1
        for i in out:
            gain[i] = -1
        i = int(gain.argmax())
        if gain[i] <= 0:
            # Every point this radius can reach is already covered. Stop
            # rather than stack: the caller places the rest by openness.
            break
        out.append(i)
        left = np.where(covers[:, i], 0.0, left)
    return out


# ── placement ────────────────────────────────────────────────────────────

def _stem(group):
    name = re.sub(r"-\d+$", "", group[0].key)
    return name if len(group) == 1 else f"{name} x{len(group)}"


def _groups(todo):
    """Identical items solved together, and the order groups are solved in.

    Biggest first, for the reason every engine here sorts that way: a 30 m
    grandstand has few legal positions and must choose before a 0.5 m bin
    takes one of them. The exception is an item something else wants to be
    `near` — the stage has to exist before the seating can face it."""
    buckets = {}
    for it in todo:
        buckets.setdefault((it.kind, it.block, round(it.w, 2), round(it.h, 2),
                            it.overhead), []).append(it)
    wanted = {w for it in todo for w in it.near}

    def rank(g):
        it = g[0]
        target = it.kind in wanted or any(i.key in wanted for i in g)
        return (0 if target else 1, -(it.w * it.h))

    return sorted(buckets.values(), key=rank)


def _dependant(later, group, free, occupied, index, step):
    """The biggest thing still to come that wants to sit beside this group.

    Only the biggest: it is the one with the fewest legal positions, so it is
    the one whose room runs out first, and reserving for it reserves for the
    smaller ones by accident."""
    keys = {it.key for it in group} | {group[0].kind}
    want = [g[0] for g in later if set(g[0].near) & keys]
    if not want:
        return None
    dep = max(want, key=lambda it: it.w * it.h)
    ground = _ground_for(free, occupied, index, dep)
    got = anchors(ground, dep.w, dep.h, step=step, rotations=dep.rotations,
                  margin=dep.clearance / 2, limit=MAX_ANCHORS,
                  ground=Ground(ground, step))
    return (dep, got) if got else None


def _place_covered(group, cands, crowd, occupied, index, notes, budget, radius):
    """Site a group by covering, then make the covering legal.

    The two halves are separate on purpose. MCLP chooses where the units
    should be, knowing about people and nothing about geometry; the loop
    afterwards moves a unit to the nearest legal candidate when the model has
    put two of them on the same six metres of tarmac. How often that fires is
    a property of the item, not of the code: none of the four toilet blocks
    ever needs it, and three of the six 15 m shade sails do, because a sail is
    big and the ground that covers the most crowd is one place. The note says
    how many moved."""
    import numpy as np
    from scipy.spatial.distance import cdist

    item = group[0]
    sub = _thin(cands, MAX_CANDIDATES,
                start=int(crowd.served([c[0] for c in cands],
                                       [c[1] for c in cands]).argmax()))
    sites = np.array([(cands[i][0], cands[i][1]) for i in sub], dtype=float)

    # Facilities of this kind already standing become extra columns, fixed
    # open. Two toilet groups (male, female) then produce one coverage of the
    # site instead of two independent ones that both hug the same plaza.
    same = [p for p in occupied
            if index.get(p.key) is not None and index[p.key].kind == item.kind]
    if same:
        sites = np.vstack([sites, np.array([(p.x, p.y) for p in same], dtype=float)])

    cost = cdist(crowd.pts[crowd.standing], sites)
    weight = crowd.weight[crowd.standing]
    if not len(cost):
        return None
    chosen, said = _cover_sites(cost, weight, len(group), len(same), radius, budget)
    notes.extend(f"{_stem(group)}: {s}" for s in said)

    order = [sub[i] for i in chosen]
    return _commit(group, order, cands, occupied, index, notes)


def _place_greedy(group, cands, crowd, occupied, index, notes, dep=None):
    """Site a group by reach, attraction, spread and room left behind.

    Deliberately the dumbest thing that works, and stated as such in the
    notes: this engine's claim is about the coverage half of the programme,
    and dressing the other half in a solver would blur which half earned the
    result."""
    import numpy as np

    item = group[0]
    xs = np.array([c[0] for c in cands])
    ys = np.array([c[1] for c in cands])
    base = crowd.served(xs, ys)
    base = base / max(base.max(), 1.0)
    if dep is not None:
        # tanh, not a normalised count: twenty places for the seating to stand
        # is as good as two hundred, and the difference that matters is
        # between some and none.
        base = base + 1.2 * np.tanh(_room(cands, item, dep[1], dep[0]) / 20.0)

    pull = np.zeros(len(cands))
    for want in item.near:
        for p in occupied:
            other = index.get(p.key)
            if other is None or (p.key != want and other.kind != want):
                continue
            d = np.hypot(xs - p.x, ys - p.y)
            pull = np.maximum(pull, np.clip(1.0 - d / NEAR_RANGE, 0, 1))

    order, done = [], []
    for _ in group:
        s = base + 1.5 * pull
        for (dx, dy, *_rest) in done:
            # Same-kind units spread rather than form a wall, unless the
            # ground gives them no choice: the penalty is soft and local.
            s -= 0.7 * np.exp(-np.hypot(xs - dx, ys - dy) / 25.0)
        s[_clash(cands, item, done)] = -np.inf
        i = int(s.argmax())
        if not np.isfinite(s[i]):
            break
        order.append(i)
        done.append((cands[i][0], cands[i][1]) + _one_size(cands[i], item)
                    + (item.clearance,))
    return _commit(group, order, cands, occupied, index, notes)


def _surplus(spare, free, crowd, occupied, index, notes, step, stem):
    """The units a covering model had no site left to open, placed by reach.

    MCLP opens at most one facility per candidate site it is offered, and only
    MAX_CANDIDATES sites are offered, so a group bigger than that comes back
    short for a reason that has nothing to do with the ground. Measured: 200
    bins on this site came back 140 placed and the engine blamed the site,
    while the identical 200 sent through the greedy placed all 200 on the same
    ground. Coverage saturates long before the two hundredth bin anyway — the
    surplus is not a covering decision, so it goes where the rest of the
    uncovered programme goes."""
    item = spare[0]
    ground = _ground_for(free, occupied, index, item)
    cands = _drop_far(anchors(ground, item.w, item.h, step=step,
                              rotations=item.rotations, margin=item.clearance / 2,
                              limit=MAX_ANCHORS, ground=Ground(ground, step)),
                      item, occupied, index)
    if not cands:
        return []
    notes.append(f"{stem}: {len(spare)} unit(s) past the {MAX_CANDIDATES} sites the "
                 "covering model is offered, so they were placed by reach")
    return _place_greedy(spare, cands, crowd, occupied, index, notes)


def _commit(group, order, cands, occupied, index, notes):
    """Turn an ordered list of candidate indices into legal placements."""
    import numpy as np

    item = group[0]
    done, taken, shifted = [], [], 0
    for idx in order:
        c = cands[idx]
        if _clash([c], item, done)[0]:
            free_idx = np.nonzero(~_clash(cands, item, done))[0]
            if not len(free_idx):
                break
            d = np.hypot(np.array([cands[i][0] for i in free_idx]) - c[0],
                         np.array([cands[i][1] for i in free_idx]) - c[1])
            c = cands[int(free_idx[int(d.argmin())])]
            shifted += 1
        done.append((c[0], c[1]) + _one_size(c, item) + (item.clearance,))
        taken.append(c)

    out = []
    for it, c in zip(group, taken):
        p = Placement(it.key, it.block, c[0], c[1], float(c[2]), label=it.label)
        out.append(p)
        occupied.append(p)
    if shifted:
        notes.append(f"{_stem(group)}: {shifted} site(s) nudged to the nearest "
                     "legal anchor — the covering model does not know footprints")
    return out


# ── the engine ───────────────────────────────────────────────────────────

def solve(items, free, region, fixed=(), items_by_key=None, seconds=20.0,
          seed=0, **_):
    """Place `items` on `free`. `seed` is accepted and ignored: nothing here
    is random, so the same site and programme give the same plan every run."""
    t0 = time.time()
    # Two lists, joined at the end. A caller — and the harness — reads the
    # first few notes and stops, so what the engine could NOT do goes at the
    # front and the model's working goes behind it.
    head, notes = [], []
    index = dict(items_by_key or {})
    index.update({it.key: it for it in items})

    placed, occupied = [], list(fixed)
    todo = []
    for it in items:
        if it.fixed:
            x, y, rot = (list(it.fixed) + [0.0])[:3]
            p = Placement(it.key, it.block, float(x), float(y), float(rot),
                          label=it.label)
            placed.append(p)
            occupied.append(p)
        else:
            todo.append(it)
    if not todo:
        return placed, ["nothing to place" if not items
                        else "every item was pinned by the brief"]

    span = max(region[2] - region[0], region[3] - region[1])
    step = max(1.0, round(span / 220, 1))

    crowd = Crowd(free, region)
    thin = len(crowd.pts) - int(crowd.standing.sum())
    notes.append(f"{len(crowd.pts)} demand points on a {DEMAND_SPACING:g} m lattice, "
                 f"{int(crowd.standing.sum())} of them with {CROWD_FLOOR:g} m2 of "
                 f"open ground within {OPEN_RADIUS:g} m"
                 + (f" — the other {thin} are slivers and carry no weight"
                    # On this site the floor drops nothing: the thinnest point
                    # still has 212 m2 open within 20 m. Saying "the rest are
                    # slivers" when there is no rest claims the weight is doing
                    # work it is not doing on this ground.
                    if thin else " — the floor excluded none of them here"))

    groups = _groups(todo)
    covered_kinds, packed = [], []
    for gi, group in enumerate(groups):
        item = group[0]
        ground = _ground_for(free, occupied, index, item)
        cands = anchors(ground, item.w, item.h, step=step, rotations=item.rotations,
                        margin=item.clearance / 2, limit=MAX_ANCHORS,
                        ground=Ground(ground, step))
        if not cands:
            head.append(f"{_stem(group)}: NOT PLACED — nowhere on this site fits "
                        f"{item.w:g}x{item.h:g} m with {item.clearance:g} m clear")
            continue
        before = len(cands)
        cands = _drop_far(cands, item, occupied, index)
        if not cands:
            head.append(f"{_stem(group)}: NOT PLACED — all {before} positions that "
                        "fit break a separation rule against something already down")
            continue

        radius = SERVICE_RADIUS.get(item.kind)
        left = seconds - (time.time() - t0)
        got = None
        if radius and crowd.standing.any() and left > 4.0:
            try:
                got = _place_covered(group, cands, crowd, occupied, index, notes,
                                     budget=min(8.0, left / 2), radius=radius)
            except Exception as e:                   # noqa: BLE001
                notes.append(f"{_stem(group)}: covering failed "
                             f"({type(e).__name__}: {e}) — placed by reach instead")
                got = None
        if got is None:
            if radius and left <= 4.0:
                notes.append(f"{_stem(group)}: out of time for a covering "
                             "model, placed by reach")
            got = _place_greedy(group, cands, crowd, occupied, index, notes,
                                dep=_dependant(groups[gi + 1:], group, free,
                                               occupied, index, step))
            packed.append(_stem(group))
        else:
            covered_kinds.append(item.kind)
            if len(got) < len(group):
                got = got + _surplus(group[len(got):], free, crowd, occupied,
                                     index, notes, step, _stem(group))
        if len(got) < len(group):
            head.append(f"{_stem(group)}: only {len(got)} of {len(group)} placed — "
                        "the ground left holds no further clear position")
        placed.extend(got)

    for kind in dict.fromkeys(covered_kinds):
        head.append(_reach(kind, placed, index, crowd))
    if packed:
        head.append("no covering model applies to these, so they were placed by "
                    "reach and spread alone: " + ", ".join(packed))
    head.insert(0, f"{len(placed)}/{len(items)} placed in {time.time() - t0:.1f}s")
    return placed, head + notes


def _reach(kind, placed, index, crowd):
    """The number this engine is answerable for: the worst walk on the site."""
    import numpy as np
    from scipy.spatial.distance import cdist

    sites = np.array([(p.x, p.y) for p in placed
                      if index.get(p.key) is not None and index[p.key].kind == kind])
    pts = crowd.pts[crowd.standing]
    if not len(sites) or not len(pts):
        return f"{kind}: nothing placed, so nothing is served"
    d = cdist(pts, sites).min(axis=1)
    i = int(d.argmax())
    target = SERVICE_RADIUS.get(kind, 0.0)
    within = 100.0 * float((d <= target).mean())
    return (f"{kind}: worst-served standing ground is ({pts[i][0]:.0f}, "
            f"{pts[i][1]:.0f}), {d[i]:.0f} m from the nearest — "
            f"{within:.0f} % of standing ground within the {target:g} m target, "
            f"median walk {np.median(d):.0f} m")


# ── validation by simulation ─────────────────────────────────────────────

MAX_AGENTS = 2000        # a ceiling; the wall-clock budget usually binds first
AGENT_RADIUS = 0.2       # m, jupedsim's default body radius
SIM_DT = 0.05            # s per step: the largest the collision-free model likes
MAX_SIM_TIME = 900.0     # s of simulated time; a site that takes longer has failed
MIN_ISLAND = 4.0         # m2 — below this a free part is drawing noise, not ground
EXIT_SIDE = 3.0          # m, the gate built around an exit point when none is given
FLOW = 1.2               # persons per metre of gate per second, the standard figure

# Wall-seconds per agent per iteration, used to decide how many agents a
# budget can afford. Measured on this site's mesh: 1.9e-4 with 600 agents all
# present, 0.2e-4 averaged over a real run, because agents leave and the
# population thins. The value below sits between the two and errs high — a
# sample that finishes early is a result, one that is cut off at the clock is
# not. The run reports what it actually cost, so the constant can be checked
# rather than trusted.
#
# The consequence is worth stating plainly: the collision-free speed model on
# a 360 m navigation mesh cannot walk a festival crowd out in thirty seconds
# of wall clock. What `egress` returns is a SAMPLE, sized to the budget, and
# every figure in the report says which of them that affects.
AGENT_STEP_COST = 1.0e-4
MESH_TOLERANCE = 0.5     # m of boundary detail dropped for the simulation mesh
MESH_MIN_HOLE = 9.0      # m2 — obstacles smaller than this are not routed around

# A kerb is not a wall. `free_space` buffers every drawn obstacle, so a bare
# LINE in the DXF — a kerb, a paving joint, the edge of a hatch — becomes a
# barrier about a metre wide, and this site's free ground arrives in 149
# disjoint parts because of it. Closing the ground by 0.6 m before simulating
# takes it to 6 parts, the largest holding 15 507 of the 15 892 m2: the
# fragmentation was almost entirely hairlines. An obstacle has to be under
# 0.2 m wide of its own to vanish at this setting, which no container, wall or
# planter is. The report says how much ground the assumption bought.
STEP_OVER = 0.6


def egress(free, placements, items_by_key, exits, headcount, seconds=30.0):
    """Walk `headcount` people out of a finished layout. Returns a report dict.

    Walkable ground is `free` minus every footprint standing on it — overhead
    objects are excluded, because a shade sail is something you walk under.
    Exits may be points or polygons; a point becomes a 3 m gate.

    Four things this had to be taught, each after it got the answer wrong:

      * jupedsim refuses a disconnected geometry outright — "accessible area
        not connected" — and this site's walkable ground is dozens of disjoint
        parts. Each part is therefore simulated separately, and the parts
        holding no exit are the finding rather than an error.
      * an exit stage must be convex and STRICTLY inside the walkable area. A
        3 m square centred 1.5 m in from the edge still has corners 2.1 m out,
        so the first version silently failed to place half the gates. The
        inset is the half-diagonal.
      * the crowd is spread over all the walkable ground, not just the part
        with a gate in it. Distributing 2 000 agents over the one exit-bearing
        island packed it to 1.8 people per square metre before the simulation
        started, and every congestion figure that came out was an artefact of
        that.
      * a density cannot be scaled with the crowd. The ground does not grow,
        so 4 people per square metre in a simulated tenth of the crowd does
        not become 40; it becomes a longer queue at the same place. Counts
        scale, densities do not, and conflating the two reported an impossible
        25 people per square metre.
      * the crowd must be sized to what the budget can finish. Taking the cap
        and stopping at the clock reported a 10-second clearance with 1 718 of
        1 799 agents still standing on the site. A small crowd that gets out
        tells the truth about routes and walking time; a large one that is cut
        off tells you nothing.

    The number worth reading is `fenced_in_by_layout`: ground a person could
    walk out of before this layout existed and cannot now. That is the
    question no bounding-box check can answer, and it is measured by running
    the same reachability twice, once on the bare site.

    Never raises. On any failure inside jupedsim it returns {"error": ...},
    because a tool that dies while checking a layout is worse than one that
    cannot check it."""
    try:
        return _egress(free, placements, items_by_key, exits, headcount, seconds)
    except Exception as e:                           # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _walkable(free, placements, items_by_key):
    """The ground a person can actually stand on and move across.

    The kerb-closing happens BEFORE the layout is subtracted, on purpose: the
    hairlines are an artefact of the drawing and should be bridged, while a
    1 m gap between two of our own food trucks is a real gap and must stay
    one."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    ground = free.buffer(STEP_OVER, join_style=2).buffer(-STEP_OVER, join_style=2)
    solid = []
    for p in placements:
        it = (items_by_key or {}).get(p.key)
        if it is None or it.overhead:
            continue
        solid.append(p.footprint(it))
    if solid:
        ground = ground.difference(unary_union(solid))
    ground = ground.buffer(0)

    parts = []
    for g in (ground.geoms if ground.geom_type.startswith("Multi") else [ground]):
        if g.geom_type != "Polygon" or g.area < MIN_ISLAND:
            continue
        # A part narrower than a person is not walkable, and it is also what
        # makes jupedsim's triangulation fail. Erode then dilate removes both
        # problems at once. Interior rings under a square metre go the same
        # way: they are drawing noise and each one costs triangles.
        core = g.buffer(-AGENT_RADIUS * 1.5).buffer(AGENT_RADIUS * 1.5)
        for q in (core.geoms if core.geom_type.startswith("Multi") else [core]):
            if q.geom_type != "Polygon" or q.area < MIN_ISLAND:
                continue
            rings = [r for r in q.interiors if Polygon(r).area >= 1.0]
            clean = Polygon(q.exterior, rings).simplify(0.05).buffer(0)
            if clean.geom_type == "Polygon" and clean.area >= MIN_ISLAND:
                parts.append(clean)
    return parts


def _point_of(spec):
    from shapely.geometry import Point
    if hasattr(spec, "geom_type"):
        return spec.centroid
    return Point(float(spec[0]), float(spec[1]))


def _gate(part, spec):
    """A convex exit stage inside `part`, or (None, 0).

    The inset is the half-DIAGONAL of the square, not its half-side. Inset by
    the half-side and the corners hang 41 % further out than the inset allows,
    `part.contains` refuses the stage, and the gate is dropped — quietly, in
    a report that then blames the layout for fencing everyone in."""
    from shapely.geometry import box
    from shapely.ops import nearest_points

    want = _point_of(spec)
    side = EXIT_SIDE
    if hasattr(spec, "geom_type"):
        side = max(spec.bounds[2] - spec.bounds[0], spec.bounds[3] - spec.bounds[1])
    for shrink in (1.0, 0.6, 0.35, 0.2):
        s = side * shrink / 2
        inner = part.buffer(-(s * math.sqrt(2) + 0.05))
        if inner.is_empty:
            continue
        c = nearest_points(inner, want)[0]
        g = box(c.x - s, c.y - s, c.x + s, c.y + s)
        if part.contains(g):
            return g, 2 * s
    return None, 0.0


def _assign(parts, specs):
    """Which walkable part each exit belongs to. -> {part index: [(gate, width)]}"""
    out, orphan = {}, []
    for spec in specs:
        want = _point_of(spec)
        # Prefer the part that contains the point; otherwise the nearest, and
        # among equals the biggest, so a gate given on a kerb line does not
        # end up opening onto the two square metres behind it.
        order = sorted(range(len(parts)),
                       key=lambda i: (round(parts[i].distance(want), 1),
                                      -parts[i].area))
        placed = False
        for i in order[:4]:
            if parts[i].distance(want) > 40.0:
                break
            g, w = _gate(parts[i], spec)
            if g is not None:
                out.setdefault(i, []).append((g, w))
                placed = True
                break
        if not placed:
            orphan.append([round(want.x, 1), round(want.y, 1)])
    return out, orphan


def _reachable(free, placements, items_by_key, specs):
    """(parts, gates, the union of ground with a gate on it)."""
    from shapely.ops import unary_union
    parts = _walkable(free, placements, items_by_key)
    gates, orphan = _assign(parts, specs)
    served = unary_union([parts[i] for i in gates]) if gates else None
    return parts, gates, orphan, served


def _egress(free, placements, items_by_key, exits, headcount, seconds):
    import jupedsim as jps

    t0 = time.time()
    specs = list(exits or [])
    if not specs:
        return {"error": "no exits given — egress needs at least one"}

    parts, gates, orphan, served = _reachable(free, placements, items_by_key, specs)
    if not parts:
        return {"error": "the layout leaves no walkable ground at all"}
    total = sum(p.area for p in parts)

    n_sim = int(min(MAX_AGENTS, max(1, headcount)))
    drawn = [g for g in (free.geoms if free.geom_type.startswith("Multi") else [free])
             if g.area >= MIN_ISLAND]
    report = {
        "walkable_m2": round(total, 1),
        "islands": len(parts),
        "islands_in_the_drawing": len(drawn),
        "kerbs_stepped_over_m": STEP_OVER,
        "headcount": int(headcount),
        "agents_simulated": 0,
        "scale": None,
        "exits_placed": sum(len(g) for g in gates.values()),
        "exits_unplaceable": orphan,
        "notes": [],
    }

    # What this layout cost, as opposed to what the site costs. The free
    # polygon comes from a drawing in which every kerb and every hatch edge is
    # an obstacle, so most of the islands were islands before anyone placed
    # anything. Running the same reachability on the bare ground separates the
    # two, and only the difference is the layout's fault.
    _, _, _, bare = _reachable(free, [], None, specs)
    lost_m2 = 0.0
    if bare is not None:
        lost = bare.difference(served) if served is not None else bare
        lost = lost.intersection(_union(parts))
        lost_m2 = lost.area
    report["fenced_in_by_layout"] = {
        "area_m2": round(lost_m2, 1),
        "agents": int(round(lost_m2 / max(total, 1e-9) * headcount)),
    }
    stranded = [p for i, p in enumerate(parts) if i not in gates]
    report["fenced_in_total"] = {
        "area_m2": round(sum(p.area for p in stranded), 1),
        "agents": int(round(sum(p.area for p in stranded)
                            / max(total, 1e-9) * headcount)),
        "islands": len(stranded),
        "at": [[round(p.centroid.x, 1), round(p.centroid.y, 1)]
               for p in sorted(stranded, key=lambda q: -q.area)[:6]],
    }
    report["notes"].append(
        "fenced_in_total counts every walkable island with no gate on it, most "
        "of which the drawing already separated with a kerb line; "
        "fenced_in_by_layout is the part this layout closed off")
    if not gates:
        report["error"] = "no exit could be placed on any walkable part"
        return report

    # Everyone is spread over ALL the walkable ground in proportion to its
    # area, which is the only defensible assumption without a crowd model of
    # the programme — and, more to the point, the only one that does not
    # invent congestion by packing the whole crowd onto whichever island
    # happens to have the gate.
    cells = {}
    sim_time, t95, left_in, cleared, cost = 0.0, 0.0, 0, True, []
    order = sorted(gates, key=lambda i: -parts[i].area)
    served_area = sum(parts[i].area for i in order)
    for i in order:
        share = parts[i].area / served_area
        budget = (seconds - (time.time() - t0)) * share
        if budget <= 1.0:
            report["notes"].append("ran out of wall clock before every island ran")
            cleared = False
            break
        # Sized to what the budget can finish, then capped by this island's
        # share of the crowd — a 40 m2 pocket does not get 200 people.
        n = min(_affordable(parts[i], gates[i], budget),
                max(1, int(round(n_sim * parts[i].area / total))))
        got = _run_part(jps, parts[i], gates[i], n, budget, cells)
        report["agents_simulated"] += got["agents"]
        sim_time = max(sim_time, got["sim_time"])
        t95 = max(t95, got["t95"])
        left_in += got["left"]
        cleared = cleared and got["cleared"]
        cost.append(got["cost"])
        report["notes"].extend(got["notes"])

    report["everyone_simulated_got_out"] = bool(cleared and left_in == 0)
    report["clearance_s"] = round(sim_time, 1)
    report["clearance_95_s"] = round(t95, 1)
    report["still_inside"] = left_in
    if cost:
        report["notes"].append(
            f"the crowd is a sample of {report['agents_simulated']}, sized so it "
            f"could finish inside the {seconds:g}s budget; the model cost "
            f"{max(cost) * 1e4:.1f}e-4 wall-seconds per agent per step here, so a "
            "denser crowd needs a bigger budget, not a faster machine")
    report["scale"] = round(headcount / max(report["agents_simulated"], 1), 1)
    report["notes"].append(
        f"the {report['agents_simulated']} agents are a sample of a crowd of "
        f"{int(headcount)}, one in {report['scale']:g}. The fenced_in figures come "
        "from ground area and not from the sample; densities are never scaled, "
        "because the ground does not grow with the crowd")
    # Travel plus flow, the standard way an egress time is built: the last
    # person's walk, then the whole crowd physically passing the gates. The
    # sample measures the first term honestly; the second is arithmetic and
    # dominates, which is itself the finding when the gates are this narrow.
    width = sum(w for g in gates.values() for _, w in g)
    queue = headcount / max(width * FLOW, 1e-6)
    walk = sim_time if report["everyone_simulated_got_out"] else t95
    report["clearance_estimate_s"] = round(walk + queue, 1)
    whose = ("walk of the last agent" if report["everyone_simulated_got_out"]
             else "walk of the 95th agent, the run not having fully cleared")
    report["notes"].append(
        f"clearance_estimate_s = the simulated {walk:.0f}s {whose} plus "
        f"{queue:.0f}s for {int(headcount)} people to pass {width:.1f} m of gate "
        f"at {FLOW:g} persons/m/s — arithmetic, not simulation")

    hot = sorted(cells.items(), key=lambda kv: -kv[1])[:5]
    density = report["agents_simulated"] / max(served_area, 1.0)
    report["congestion"] = [{"at": [k[0], k[1]], "peak_per_m2": round(v / 16.0, 2)}
                            for k, v in hot]
    report["congestion_is_meaningful"] = bool(density >= 0.05)
    if not report["congestion_is_meaningful"]:
        report["notes"].append(
            f"the sample stands at {density:.3f} people per m2, far below any real "
            "crowd, so `congestion` marks where the routes converge and not how "
            "bad the crush will be; what this run measures is whether the routes "
            "exist and how long the walk is")
    report["seconds"] = round(time.time() - t0, 1)
    return report


def _union(parts):
    from shapely.ops import unary_union
    return unary_union(parts)


def _mesh(part):
    """The island as jupedsim should see it, which is not how shapely holds it.

    Two departures from the exact ground, both bought deliberately:
    half a metre of boundary detail, and obstacles under 9 m2. A person walks
    round a bin without routing around it, and making the model route around
    one costs four times the wall clock — measured, on this island: 590
    boundary points and 36 holes ran at 7.4e-4 wall-seconds per agent-step,
    189 points and 22 holes at 1.9e-4.

    Reachability — the finding this whole function exists for — is measured on
    the exact ground in `_walkable`, not here, so the simplification cannot
    hide a layout that fences the crowd in."""
    from shapely.geometry import Polygon
    s = part.simplify(MESH_TOLERANCE)
    rings = [r for r in s.interiors if Polygon(r).area >= MESH_MIN_HOLE]
    m = Polygon(s.exterior, rings).buffer(0)
    if m.geom_type != "Polygon":
        m = max(m.geoms, key=lambda g: g.area)
    return m


def _affordable(part, gates, budget):
    """How many agents this island can actually walk out inside `budget`.

    The alternative — take the cap, start, and stop when the clock runs out —
    was the first version, and it reported a 10-second clearance time with
    1 718 of 1 799 agents still standing on the site. A small crowd that
    finishes tells you the truth about routes and travel time; a large one
    that is cut off tells you nothing at all."""
    far = 1.0
    for x, y in part.exterior.coords:
        d = min(math.hypot(x - g.centroid.x, y - g.centroid.y) for g, _ in gates)
        far = max(far, d)
    # 1.5x the straight line for going round things, plus half a minute of
    # queueing at the gate.
    iters = (1.5 * far / 1.2 + 30.0) / SIM_DT
    return int(max(12, min(MAX_AGENTS, budget / (AGENT_STEP_COST * iters))))


def _run_part(jps, part, gates, n, budget, cells):
    """One connected island: distribute agents, send each to its nearest gate."""
    out = {"agents": 0, "left": 0, "sim_time": 0.0, "t95": 0.0, "cleared": True,
           "notes": [], "cost": 0.0}
    part = _mesh(part)
    sim = jps.Simulation(model=jps.CollisionFreeSpeedModel(), geometry=part, dt=SIM_DT)
    stages = []
    for g, _w in gates:
        sid = sim.add_exit_stage(list(g.exterior.coords)[:-1])
        stages.append((sid, sim.add_journey(jps.JourneyDescription([sid])),
                       g.centroid))

    pts = []
    for attempt in (n, n // 2, n // 4, n // 8):
        if attempt < 1:
            break
        try:
            pts = jps.distribute_by_number(
                polygon=part, number_of_agents=attempt,
                distance_to_agents=0.55, distance_to_polygon=AGENT_RADIUS + 0.1,
                seed=1)
            break
        except Exception:                    # noqa: BLE001 - island too small to fill
            pts = []
    for p in pts:
        # Each agent heads for the gate nearest to it. jupedsim can express
        # "whichever door is emptiest" with a transition, but a crowd on an
        # open site walks to the gate it can see, and modelling perfect
        # knowledge of queue lengths would flatter the layout.
        sid, jid, _c = min(
            stages, key=lambda s: (s[2].x - p[0]) ** 2 + (s[2].y - p[1]) ** 2)
        try:
            sim.add_agent(jps.CollisionFreeSpeedModelAgentParameters(
                position=p, journey_id=jid, stage_id=sid))
            out["agents"] += 1
        except Exception:                            # noqa: BLE001 - overlapping start
            pass
    if not out["agents"]:
        out["notes"].append(f"island at ({part.centroid.x:.0f}, {part.centroid.y:.0f}) "
                            "took no agents")
        return out

    t0 = time.time()
    every = int(round(1.0 / SIM_DT))
    it = 0
    # The time by which 95 % are out, kept separately from the time the last
    # one is out. A single agent wedged in a corner — one in 598 on this site
    # — runs the clock to the cap and makes a 400-second site look like a
    # 900-second one. Both numbers are reported; neither is allowed to stand
    # in for the other.
    left_95 = max(1, int(math.ceil(out["agents"] * 0.05)))
    while sim.agent_count() > 0:
        sim.iterate()
        it += 1
        if it % every == 0:
            _tally(sim, cells)
            if not out["t95"] and sim.agent_count() <= left_95:
                out["t95"] = sim.elapsed_time()
        if it % 20 == 0 and (time.time() - t0 > budget
                             or sim.elapsed_time() > MAX_SIM_TIME):
            break
    out["cost"] = (time.time() - t0) / max(it * out["agents"], 1)
    out["sim_time"] = sim.elapsed_time()
    out["t95"] = out["t95"] or sim.elapsed_time()
    out["left"] = sim.agent_count()
    out["cleared"] = sim.agent_count() == 0
    if out["left"]:
        out["notes"].append(
            f"island at ({part.centroid.x:.0f}, {part.centroid.y:.0f}): {out['left']} "
            f"of {out['agents']} agents still inside after {out['sim_time']:.0f}s")
    return out


def _tally(sim, cells):
    """Peak occupancy per 4 m cell over the whole run.

    4 m cells because a density measured over less than that is two people
    walking side by side, not congestion."""
    now = {}
    for a in sim.agents():
        x, y = a.position
        k = (math.floor(x / 4) * 4 + 2.0, math.floor(y / 4) * 4 + 2.0)
        now[k] = now.get(k, 0) + 1
    for k, v in now.items():
        if v > cells.get(k, 0):
            cells[k] = v
