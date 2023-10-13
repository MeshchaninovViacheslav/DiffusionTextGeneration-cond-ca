from subprocess import run, STDOUT, PIPE, Popen


p1 = Popen(["ps", "aux"], stdout=PIPE)
p2 = Popen(["grep", "/home/vmeshchaninov/.conda/envs/fap2_env/bin/python*"], stdin=p1.stdout,
           stdout=PIPE, text=True)

p1.stdout.close()

output = p2.communicate()[0]

pids = []

for line in output.strip().split("\n"):
    for i in range(10):
        line = line.replace("  ", " ")
    print(line.split(" ")[1])
    pids.append(line.split(" ")[1])

for pid in pids:
    Popen(["kill", "-SIGKILL", pid])
