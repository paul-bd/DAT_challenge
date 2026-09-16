"""Count MAIN datscan trainer processes (DataLoader workers share the command line; exclude by PPID)."""
import subprocess, sys
sub = sys.argv[1] if len(sys.argv) > 1 else ""      # optional substring filter, e.g. an --out dir (amel5 counts only its own)
out = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout.splitlines()[1:]
rows = [l.split(None, 2) for l in out]
ok = lambda a: "python -m datscan --member" in a and sub in a and "nohup" not in a and "for " not in a   # exclude launcher shells whose command line quotes the trainer invocation (10th self-match)
pids = {r[0] for r in rows if len(r) == 3 and ok(r[2])}
print(sum(1 for r in rows if len(r) == 3 and ok(r[2]) and r[1] not in pids))
