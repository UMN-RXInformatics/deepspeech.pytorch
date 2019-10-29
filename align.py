import sys
import os
import re
import argparse
from sox import file_info
import json
import Levenshtein
import subprocess
import time
import signal

# poor-man's norm
def norm(txtin):


  # remove things in parentheses
  txtout = re.sub(r'\([^\)]+\)','',txtin)

  # remove tags
  txtout = re.sub(r'FILLEDPAUSE\_','', txtout, flags=re.IGNORECASE)
  txtout = re.sub(r'T\_[^\s+]+','', txtout, flags=re.IGNORECASE)
  txtout = re.sub(r'E\_', '', txtout, flags=re.IGNORECASE)

  # remove noise
  txtout = re.sub(r'\&\S+(\s+|$)', ' ', txtout)
  txtout = re.sub(r'\[\%\s+[^\%]+\]', ' ', txtout)
  txtout = re.sub(r'(\s+|^)X+(\s+|$)', ' ', txtout)
  txtout = re.sub(r'(\s+|^)x+(\s+|$)', ' ', txtout)
  txtout = re.sub(r'\@\S+(\s+|$)', ' ', txtout) # @fp or @i ..
  txtout = re.sub(r'\*[A-Z]+\:', ' ', txtout)

  # convert uh to ah and uhm to um
  txtout = re.sub(r'(\s+|^)uh(\s+|$)', ' ah ', txtout)
  txtout = re.sub(r'(\s+|^)uhm(\s+|$)', ' um ', txtout)

  txtout = re.sub(r'[\(|\[][^\(|\[]+[\)|\]]', '', txtout)
  txtout = re.sub(r'[^a-zA-Z \']','',txtout)
  txtout = re.sub(r'\s+', ' ', txtout)
  return re.sub(r'\n',' ',txtout.upper())


def chunk_it(wavfile, strs, characters, options, basename, size):


    # make chunks be around [size] seconds long - skip chunking if file is shorter than [size]
    dur = file_info.duration(wavfile)
    print("Duration: "+str(dur)+" Max size: "+str(size))
    if dur <= size:
        chunknum = 1
        wavechunkout = options.output + "/" + basename + "_" + str(chunknum) + ".wav"
        txtchunkout = open(options.output + "/" + basename + "_" + str(chunknum) + ".txt", "w")
        txtchunkout.write(''.join(characters))
        txtchunkout.close()
        # print(basename+ " "+str(newstrs[i]) + " " + str(newstps[i]) + " " + str(length) + " " +str(chunknum))
        os.system('sox ' + wavfile + ' ' + wavechunkout + ' channels 1 rate 16000')
        #manifestfile.write(wavechunkout + "," + options.output + "/" + basename + "_" + str(chunknum) + ".txt\n")

        return

    newstrs = strs
    newtexts = characters


    #manifestfile = open(options.omanifest, "w")
    chunknum = 1
    chunktext = ""
    nextchunkstart = 0
    currlength = 0

    for i in range(1, len(newstrs)):

        segmentlength = newstrs[i] - newstrs[i-1]
        currlength = newstrs[i]

        if currlength - newstrs[nextchunkstart] >= size and segmentlength > 0.5 and characters[i-1] == " ": # > a silent pause included
            # move back 100 ms from the start of the next word
            splitpoint = newstrs[i] - 0.1
            splitpointix = i # excluding i

            #print(str(strs[i-1]) +" "+ str(strs[i]) +" "+ characters[i-1]+" "+ characters[i]+" "+ characters[i+1]+" "+ characters[i+2])

            #  write out to file
            wavechunkout = options.output+"/"+basename+"_"+str(chunknum)+".wav"
            txtchunkout = open(options.output + "/" + basename + "_" + str(chunknum) + ".txt","w")
            txtchunkout.write(''.join(newtexts[nextchunkstart:splitpointix-1]))
            txtchunkout.close()

            audiochunklen = newstrs[splitpointix] - newstrs[nextchunkstart] - 0.1

            #print(basename+ " "+str(newstrs[i]) + " " + str(newstps[i]) + " " + str(length) + " " +str(chunknum))
            os.system('sox ' + wavfile + ' ' + wavechunkout + ' channels 1 rate 16000 trim ' + str(newstrs[nextchunkstart]) + ' ' + str(audiochunklen))
            #manifestfile.write(wavechunkout + "," + options.output + "/" + basename + "_" + str(chunknum) + ".txt\n")
            chunknum = chunknum+1
            nextchunkstart = i
            currlength = 0

    # last chunk
    #  write out to file
    wavechunkout = options.output + "/" + basename + "_" + str(chunknum) + ".wav"
    txtchunkout = open(options.output + "/" + basename + "_" + str(chunknum) + ".txt", "w")
    txtchunkout.write(''.join(newtexts[nextchunkstart:len(newstrs)]))
    txtchunkout.close()
    # print(basename+ " "+str(newstrs[i]) + " " + str(newstps[i]) + " " + str(length) + " " +str(chunknum))
    os.system('sox ' + wavfile + ' ' + wavechunkout + ' channels 1 rate 16000 trim ' + str(newstrs[nextchunkstart]))
    #manifestfile.write(wavechunkout + "," + options.output + "/" + basename + "_" + str(chunknum) + ".txt\n")

    #manifestfile.close()

def align(wav, txt, wdir, basename, lm=None):

    transcript_arr = None
    offsets_arr = None
    langmod = None

    if lm == None:
        # normalize text
        txtin = open(txt,"r")
        txtout = open(wdir+"/"+basename+".txt","w")
        text = txtin.readline()
        nrmtxt = norm(text)
        txtout.write(nrmtxt+"\n")
        txtout.close()

        # build KenLM
        #os.system("/home/pakh0002/kenlm/build/bin/lmplz -o 6 --discount_fallback --text " + wdir + "/" + basename + ".txt " + "--arpa "+ wdir + "/" + basename + ".arpa")
        #os.system("/home/pakh0002/kenlm/build/bin/build_binary "+ wdir + "/" + basename + ".arpa " + wdir + "/" + basename + ".binary")
        os.system("/workspace/kenlm/build/bin/lmplz -o 6 --discount_fallback --text " + wdir + "/" + basename + ".txt " + "--arpa " + wdir + "/" + basename + ".arpa")
        os.system("/workspace/kenlm/build/bin/build_binary "+ wdir + "/" + basename + ".arpa " + wdir + "/" + basename + ".binary")
        langmod = wdir + "/" + basename + ".binary"
    else:
        langmod = lm

    # convert audio
    os.system("sox "+wav+" -r 16000 -c 1 "+wdir + "/" + basename + "_16.wav" )

    # run recognizer
    print("Recognizing WAV... "+wav)
    inwav = wdir + "/" + basename + "_16.wav"
    outjson = wdir + "/" + basename + ".json"
    os.system("curl -X POST http://0.0.0.0:8888/transcribe -H \"Content-type: multipart/form-data\" -F \"file=@"+wav+"\" > "+outjson)

    #os.system("python ./transcribe.py --audio-path " +
    #          wdir + "/" + basename + "_16.wav --model-path /home/pakh0002/deepspeech.pytorch/expmodels/deepspeech_100g_cv.pth.tar --lm-path " +
    #          langmod + " --decoder beam --alpha 0.9 --cuda --offsets > " + wdir + "/" + basename + ".json");

    # read JSON
    with open(wdir + "/" + basename + ".json") as f:
        asr_data = json.load(f)

    transcript = asr_data["output"][0]["transcription"]
    offsets_arr = asr_data["output"][0]["offsets"]

    # take care of cases where the transcription is blank and no offsets
    if transcript == "":
        transcript = "\'"
        offsets_arr.append(0)

    transcript_arr = list(transcript)

    assert len(offsets_arr) == len(transcript_arr), "Numbers of characters and offsets in output do not match"

    # multiply the offsets by a scalar (duration of file in seconds / size of output) to get the offsets in seconds
    dur = file_info.duration(wdir + "/" + basename + "_16.wav")
    #coeff =  float(dur) / 16000.0
    offsets_arr_secs = [round(i * 0.02,2) for i in offsets_arr]

    #print(len(offsets_arr))
    #print(len(transcript_arr))
    print("Result: " + transcript)

    return [offsets_arr_secs,transcript_arr,wdir + "/" + basename + "_16.wav"]

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--imanifest', help='Input manifest', default='./')
  parser.add_argument('--output', help='Output dir', default='./')
  parser.add_argument('--omanifest', help='Output manifest', default = './out.txt')
  parser.add_argument('--chunksize', help='Size of audio segments (seconds)', default=8.0)
  parser.add_argument('--wdir', help='Working directory', default='./')

  args = parser.parse_args()

  inmani = open(args.imanifest,"r")

  for line in inmani.readlines():
      line = re.sub(r'\s+','',line)
      wavefile,transcript = line.split(",")

      if wavefile.endswith(".wav"):

          basename = os.path.basename(wavefile)
          dirname = os.path.dirname(wavefile)
          basename = re.sub(r'\.wav', '', basename)

          if os.path.isfile(wavefile) and os.path.isfile(transcript):
              if file_info.duration(wavefile) > 20:
                  # split the long wavefile
                  os.system("sox -V3 "+wavefile+" "+dirname+"/"+basename+"_part_.wav silence -l 0 1 2.0 0.3\% : newfile : restart")
                  partcnt = 0
                  for part in os.listdir(dirname):
                      if "_part_" in part:
                          partcnt = partcnt + 1

                  if partcnt < 3:
                      os.system("rm -rf "+dirname+"/*_part_*")
                      # decrease silence dur to 1 second
                      os.system("sox -V3 " + wavefile + " " + dirname + "/" + basename + "_part_.wav silence -l 0 1 2.0 2\% : newfile : restart")

                  partcnt = 0
                  for part in os.listdir(dirname):
                      if "_part_" in part:
                          partcnt = partcnt + 1

                  if partcnt < 3:
                      os.system("rm -rf " + dirname + "/*_part_*")
                      # decrease silence dur to 1 second
                      os.system(
                          "sox -V3 " + wavefile + " " + dirname + "/" + basename + "_part_.wav silence -l 0 1 2.0 5\% : newfile : restart")

                  combined_wavs = {}

                  # build an LM
                  # normalize text
                  txtin = open(transcript, "r")
                  txtout = open(args.wdir + "/" + basename + ".txt", "w")
                  text = txtin.readline()
                  nrmtxt = norm(text)
                  txtout.write(nrmtxt + "\n")
                  txtout.close()

                  # build KenLM
                  os.system("/workspace/kenlm/build/bin/lmplz -o 6 --discount_fallback --text " + args.wdir + "/" + basename + ".txt " + "--arpa " + args.wdir + "/" + basename + ".arpa")
                  os.system("/workspace/kenlm/build/bin/build_binary " + args.wdir + "/" + basename + ".arpa " + args.wdir + "/" + basename + ".binary")
                  lmname = args.wdir + "/" + basename + ".binary"

                  # start the server

                  subproc = subprocess.Popen(['exec python server.py --model-path /home/pakh0002/deepspeech.pytorch/expmodels/deepspeech_100g_cv.pth.tar --lm-path ' + args.wdir + '/' + basename + '.binary --decoder beam --alpha 0.9 --cuda'], shell=True)
                  time.sleep(20)
                  pid = subproc.pid
                  print("Started ASR server process: "+str(pid))
                  #inwav = '/media/pakh0002/AUDIO_DATA_SSD/SWC_English/english/2006_Atlantic_hurricane_season/chunks/audio_part_001_1.wav'
                  #os.system("curl -X POST http://0.0.0.0:8888/transcribe -H \"Content-type: multipart/form-data\" -F \"file=@"+inwav+"\"")


                  for subname in os.listdir(dirname):

                      if "_part_" in subname:

                          partbasename = subname
                          partbasename = re.sub(r'\.wav','',partbasename)

                          starts, transcrs, wav = align(dirname+"/"+subname, transcript, args.wdir, partbasename, lmname)
                          chunk_it(wav, starts, transcrs, args, partbasename, float(args.chunksize))
                          #combined_wavs[wav] = [starts, transcrs]
                          #for wav in combined_wavs:
                          #print("WAVE to chunk: "+wav)
                          #chunk_it(wav, combined_wavs[wav][0], combined_wavs[wav][1], args, partbasename, float(args.chunksize))

                  #os.killpg(os.getpgid(pid), signal.SIGTERM)
                  subproc.kill()
                  print("Ended ASR server process: " + str(pid))

              else:

                  # build an LM
                  # normalize text
                  txtin = open(transcript, "r")
                  txtout = open(args.wdir + "/" + basename + ".txt", "w")
                  text = txtin.readline()
                  nrmtxt = norm(text)
                  txtout.write(nrmtxt + "\n")
                  txtout.close()

                  # build KenLM
                  os.system("/workspace/kenlm/build/bin/lmplz -o 6 --discount_fallback --text " + args.wdir + "/" + basename + ".txt " + "--arpa " + args.wdir + "/" + basename + ".arpa")
                  os.system("/workspace/kenlm/build/bin/build_binary " + args.wdir + "/" + basename + ".arpa " + args.wdir + "/" + basename + ".binary")
                  lmname = args.wdir + "/" + basename + ".binary"

                  # start the server
                  subproc = subprocess.Popen(['exec python server.py --model-path /home/pakh0002/deepspeech.pytorch/expmodels/deepspeech_100g_cv.pth.tar --lm-path ' + args.wdir + '/' + basename + '.binary --decoder beam --alpha 0.9 --cuda'],shell=True)
                  time.sleep(20)
                  pid = subproc.pid
                  print("Started ASR server process: " + str(pid))

                  starts, transcrs, wav = align(wavefile, transcript, args.wdir, basename, lmname)
                  chunk_it(wav, starts, transcrs, args, basename, float(args.chunksize))


                  subproc.kill()
                  print("Ended ASR server process: " + str(pid))

          continue
      else:
          continue



