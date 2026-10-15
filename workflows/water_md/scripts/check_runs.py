import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water import D, STRIDE, read_properties, read_trajectory  # noqa: E402


def checkpoint_step(path):
    if not path.exists() or path.stat().st_size == 0:
        return None

    found = re.search(r"<step>\s*(\d+)\s*</step>", path.read_text())
    return int(found.group(1)) if found else 0


def last_property_step(run):
    if not (run / "sim.out").exists():
        return 0

    props, _ = read_properties(run)
    return int(props["step"][-1])


# completion of each run from its own trajectory files, independent of the job state
def check(run, total_steps):
    rec = {"run": run.name}
    if not (run / "sim.out").exists():
        rec["state"] = "not started"
        return rec, False

    props, prop_check = read_properties(run)
    traj, frame_check = read_trajectory(run)
    log = (run / "segments.log").read_text() if (run / "segments.log").exists() else ""

    rec.update(
        property_last_step=int(props["step"][-1]),
        last_paired_frame_step=int(traj["steps"][-1]) if len(traj["steps"]) else None,
        paired_frames=frame_check["paired_frames"],
        expected_frames=total_steps // STRIDE + 1,
        restart_step=checkpoint_step(run / "RESTART"),
        periodic_checkpoint_step=checkpoint_step(run / "sim.chk"),
        segments=log.count("start from"),
        ipi_exits=[int(x) for x in re.findall(r"i-PI exit (\d+)", log)],
        refused_restarts=log.count("not restarted"),
        unreadable_checkpoints_set_aside=log.count("unreadable, set aside"),
        property_check=prop_check,
        frame_check=frame_check,
    )

    complete = (
        rec["property_last_step"] >= total_steps
        and rec["last_paired_frame_step"] == total_steps
        and rec["paired_frames"] == rec["expected_frames"]
        and not frame_check["missing_steps"]
        and not frame_check["unexpected_steps"]
        and not frame_check["steps_not_in_all_three"]
    )
    rec["state"] = "complete" if complete else "incomplete"
    return rec, complete


# usage check_runs.py phase, with --incomplete or --steps for the fast property-only forms used by the job chain
def main():
    phase = sys.argv[1]
    names = sorted(p.name for p in (D / "runs" / phase).glob("*_seed*"))
    total = json.loads((D / "runs" / phase / names[0] / "run.json").read_text())["steps"]

    # the summed completed steps over all runs, a measure of progress between job segments
    if "--steps" in sys.argv:
        print(sum(last_property_step(D / "runs" / phase / name) for name in names))
        return

    if "--incomplete" in sys.argv:
        for name in names:
            if last_property_step(D / "runs" / phase / name) < total:
                print(name)
        return

    all_complete = True
    records = []

    for name in names:
        rec, complete = check(D / "runs" / phase / name, total)
        all_complete = all_complete and complete
        records.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("property_check", "frame_check")}))

    (D / "analysis").mkdir(exist_ok=True)
    (D / "analysis" / f"completion_{phase}.json").write_text(json.dumps(records, indent=1) + "\n")

    print("ALL COMPLETE" if all_complete else "NOT ALL COMPLETE")
    raise SystemExit(0 if all_complete else 1)


if __name__ == "__main__":
    main()
