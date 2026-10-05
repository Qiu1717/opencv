import os
root = r"D:\计算机视觉\qzz"
junk = ["alive.txt","cd.txt","cleanup_done.txt","e.txt","ok3.txt","pid.txt","port.txt",
        "ppid.txt","proc.txt","st.txt","v2.txt","v3.txt","v4.txt","fin.txt",
        "run_server.ps1","run.log","run_err.log","pid.txt"]
n = 0
for f in junk:
    p = os.path.join(root, f)
    if os.path.isfile(p):
        try:
            os.remove(p); n += 1
        except Exception:
            pass
open(os.path.join(root, "cdone.txt"), "w", encoding="utf-8").write("removed %d\n" % n)