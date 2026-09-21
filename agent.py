"""plan2event-cover — coverage siting, checked by crowd simulation.

The engine is engines/cover.py, after pysal/spopt for location-allocation and
PedestrianDynamics/jupedsim for validation. Every other engine answers "does
it fit"; this one answers "can everyone reach it" — the binding question for
toilets, water, first aid and exits — and then simulates the crowd leaving to
find the layouts that fence people in.

Run:    python agent.py
Deploy: python agent.py deploy
"""

import asyncio
import pathlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cycls

from core import wiring

# A Windows host pickles WindowsPath, which a Linux container cannot rebuild.
pathlib.WindowsPath.__reduce__ = lambda self: (pathlib.PurePosixPath, (self.as_posix(),))

IMAGE = wiring.image("spopt", "geopandas", "pulp", "jupedsim")
# This repository is the v2 variant. v1 and v2 of an engine are the same code;
# v2 additionally reads reference/, carries the composition rules, and is
# reviewed and revised twice after it stops.
V2 = True
NAME = "plan2event-cover" + ("-v2" if V2 else "")


@cycls.agent(name=NAME, image=IMAGE, web=wiring.web("Event plan — coverage and egress"),
             memory="4Gi", volumes=wiring.volumes(NAME))
async def plan2event_cover(context):
    from core import prompt, sdk, tools
    from engines import cover

    ws = Path(context.workspace.root)
    ws.mkdir(parents=True, exist_ok=True)
    tools.intake(context, ws)

    if not (ws / tools.PLAN).exists():
        yield ("Attach the venue plan as a **DXF** and describe the event — "
               "what it is, who comes, how many, and anything the site or the "
               "country requires.")
        return

    session = tools.Session(ws, cover.solve)
    specs = tools.schemas(cover.NAME, cover.DOC,
                          getattr(cover, "CONSTRAINT_HELP", ""))

    async for event in sdk.drive(
            context, ws, tools.make_handlers(session), specs,
            prompt.build(cover.NAME, cover.DOC,
                         takes_constraints=hasattr(cover, "parse"),
                         rounds=2 if V2 else 0, knowledge=V2),
            session=session, rounds=2 if V2 else 0):
        yield event

    if (ws / tools.PLAN).exists():
        await asyncio.to_thread(shutil.copy, ws / tools.PLAN, ws / tools.OUT)
        yield (f"\n\nSaved as **{tools.OUT}** — download it from the Files "
               f"panel. Everything added is on `EVENT-*` layers; deleting them "
               f"returns the drawing you sent.")


wiring.finalise(plan2event_cover)

if __name__ == "__main__":
    plan2event_cover.deploy() if "deploy" in sys.argv else plan2event_cover.local()
