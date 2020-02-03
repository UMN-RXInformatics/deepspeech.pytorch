from subprocess import Popen, PIPE, STDOUT
import sys
import os
import shutil

if len(sys.argv) != 3 :
    print "usage:",sys.argv[0],"<audiofile> ","<samplerate>"
    exit(1)

debug = False

CHUNKSIZE=8192

def kaldidump(fname):
    f = open(fname,'rb')
    while True:
        chunk = f.read(CHUNKSIZE)
        if chunk:
            os.write(sys.stdout.fileno(),chunk)
        else:
            break;


def execute(cmd):
    if debug: print cmd
    p = Popen(cmd,stdout=PIPE,stderr=STDOUT)  # merge stdin/stdout
    return p.communicate()

infile = sys.argv[1]
rate = str(sys.argv[2])
outfile = "max" + infile

cmd = ["/usr/bin/sox"]
cmd.append(infile)
cmd.append("-n")
cmd.append("stat")
cmd.append("-v")

if debug: print cmd
scale = execute(cmd)[0].strip()

print(scale)

sf = 0.85 * float(scale)
if debug: print infile,"scale factor",sf

if sf > 1.0:
    cmd = ["/usr/bin/sox"]
    cmd.append("-v")
    cmd.append("%f" % sf)
    cmd.append(infile)
    cmd.append('-c')
    cmd.append('1')
    cmd.append('-r')
    cmd.append(rate)
    cmd.append('-t')
    cmd.append('wav')
    #cmd.append('-')
    cmd.append(outfile)
    res =  execute(cmd)
    if debug: print res
    os.write(sys.stdout.fileno(),res[0])
else:
    if debug:  print "no change",infile
#shutil.copyfile(infile,outfile)
    kaldidump(infile)

