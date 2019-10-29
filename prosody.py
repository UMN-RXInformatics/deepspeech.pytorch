import argparse
import warnings
import sys
import re

from opts import add_decoder_args, add_inference_args
from utils import load_model

warnings.simplefilter('ignore')

from decoder import GreedyDecoder

import torch

from data.data_loader import SpectrogramParser
from model import DeepSpeech
import os.path
import json
import textgrid


def correct_asr(transcr_text, asr_output):

    # after alighning, let's correct the ASR output

    # create a reference and a hypothesis files
    ref = transcr_text
    reftextarr = ["<S>"]
    r = open(ref,"r")
    for ln in r:
        for w in ln.split():
            reftextarr.append(w)
    r.close()



    # run sclite
    print("HYP " + "(" + str(len(hyptextarr)) + "): " + hyp)
    print("REF " + "(" + str(len(reftextarr)) + "): " + ref)

    diff = 0
    # make sure REF and HYP are the same size
    if len(hyptextarr) < len(reftextarr):
        diff = len(reftextarr) - len(hyptextarr)
        for x in range(0, diff - 1):
            hyptextarr.append("XXXXX ")
    elif len(hyptextarr) > len(reftextarr):
        diff = len(hyptextarr) - len(reftextarr)
        for x in range(0, diff - 1):
            reftextarr.append("XXXXX ")

    hyptextarr.append("</S>")
    reftextarr.append("</S>")

    r = open(ref,"w")
    r.write(' '.join(reftextarr)+"\n")
    r.close()
    h = open(hyp, "w")
    h.write(' '.join(hyptextarr) + "\n")
    h.close()

    # run the alignment
    os.system("/home/pakh0002/sclite/sclite -r "+ref+" -h "+ hyp + " -i wsj -o all")

    # parse sclite output
    sc = open(wdir+"/audio_asr.txt.pra")
    lcnt = 0
    refarray = []
    hyparray = []
    statusarr = []

    for line in sc:
        line = re.sub(r"\>\>\s+","",line)
        line = line.strip()
        line = re.sub(r'\t','|',line)
        #print(line)
        if line.startswith("REF:"):
            line = re.sub(r'REF: ','',line)
            for e in line.split():
                refarray.append(e)
        if line.startswith("HYP:"):
            line = re.sub(r'HYP: ', '', line)
            for e in line.split():
                hyparray.append(e)
        if line.startswith("Eval:"):
            line = re.sub(r'Eval: ', '', line)
            for e in [ch for ch in line]:
                statusarr.append(e)
    sc.close()

    


def convert_to_rec(decoded_results):

    if decoded_results != None:

        transcr_wrd_starts = []
        transcr_wrd_ends = []
        transcr_wrd_txt = []
        transcr_ph_starts = []
        transcr_ph_ends = []
        transcr_ph_txt = []

        
        transcr = decoded_results["output"][0]["transcription"]
        offs = decoded_results["output"][0]["offsets"]
        
        #print(offs)

        for ix in range(0,len(offs)):
            offs[ix] = (float(offs[ix]*20)/float(1000))+0.20

        #print(offs) 

        tmp_transcr = []
        tmp_offs = []
        silflag = 0
        ix = 0
        # merge multiple silences in a row
        while ix < len(transcr):
            #print(str(ix)+" "+str(len(transcr)))
            while transcr[ix] == " " and ix < len(transcr)-1:
                silflag = 1
                ix += 1
    
            if silflag == 1:
                tmp_transcr.append(" ")
                tmp_offs.append(offs[ix-1])
                silflag = 0
                continue  

            tmp_transcr.append(transcr[ix])
            tmp_offs.append(offs[ix])
            ix += 1

        transcr = tmp_transcr
        offs = tmp_offs
  
        
        transcr_wrd_starts.append(0)
        wrd = ""
        for ix in range(0,len(transcr)-1):


            transcr_ph_starts.append(offs[ix])
            transcr_ph_ends.append(offs[ix+1])
            transcr_ph_txt.append(transcr[ix])
  
            print(transcr[ix] + " " + str(offs[ix]))

            # silent pause
            if transcr[ix] == " ":
                transcr_wrd_ends.append(offs[ix])
                if wrd == "@": wrd = "FILLEDPAUSE_um"
                transcr_wrd_txt.append(wrd)
       
                transcr_wrd_txt.append("SIL")
                transcr_wrd_starts.append(offs[ix])
                transcr_wrd_ends.append(offs[ix+1])

                wrd = ""
                transcr_wrd_starts.append(offs[ix+1])  
            else:
                if ix == 0:  transcr_wrd_starts[ix] = offs[ix]
                wrd = wrd + transcr[ix]



    return [transcr_wrd_starts, transcr_wrd_ends, transcr_wrd_txt, transcr_ph_starts, transcr_ph_ends, transcr_ph_txt]

def decode_results(model, decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def transcribe(audio_path, parser, model, decoder, device):
    spect = parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets

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

  txtout = re.sub(r'(\s+|^)(UM|AH)(\s+|$)', ' @ ', txtout)

  return re.sub(r'\n',' ',txtout.upper()) 




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser = add_inference_args(parser)
    parser.add_argument('--audio-path', default='audio.wav',help='Audio file to predict on')
    parser.add_argument('--outputfile', default='out.json',help='Output file')
    parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
    parser.add_argument('--outformat', default='json', help='Output format (json|textgrid)')
    parser.add_argument('--align', type=bool,default=False, help='Do alignment from a transcript')
    parser.add_argument('--transcript', default='./trans.txt', help='Do alignment from a transcript')
    parser.add_argument('--tempdir', default='/temp/', help='Temporary folder')


    parser = add_decoder_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
 

    basename = os.path.basename(args.transcript)
    basename = re.sub(r'\.txt','',basename)
    basedir = os.path.dirname(args.transcript)



    if args.align == True:
        txtin = open(args.transcript, "r")
        txtout = open(args.transcript+".tmp", "w")
        text = txtin.readline()
        nrmtxt = norm(text)
        txtout.write(nrmtxt)
        txtout.close()
        
        # create manifest
        with open(args.transcript+"_manifest.csv","w") as man:
            man.write(args.audio_path+","+args.transcript+".tmp\n")

        

        os.system("python /workspace/deepspeech.pytorch/train.py --cuda --train-manifest " + args.transcript+"_manifest.csv " + "--val-manifest " + args.transcript+"_manifest.csv " + "--model-path " + args.tempdir+"/"+basename+".pth.tar " + " --save-folder "+args.tempdir+ " --epochs 50 --batch-size 1 --num-workers 0 --finetune --continue-from "+args.model_path)


        if os.path.exists(args.tempdir+"/"+basename+".pth.tar"):
            model = load_model(device, args.tempdir+"/"+basename+".pth.tar", args.cuda)
        else:
            print("ERROR! Could not find model "+args.tempdir+"/"+basename+".pth.tar ")
            sys.exit()
    else:
        model = load_model(device, args.model_path, args.cuda)


 

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=lmpath, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))


    parser = SpectrogramParser(model.audio_conf, normalize=True)

    decoded_output, decoded_offsets = transcribe(args.audio_path, parser, model, decoder, device)
    decoded_results = decode_results(model, decoded_output, decoded_offsets)
    endpoints = convert_to_rec(decoded_results)


    if args.align == True:

        txtin = open(args.transcript, "r")
        txtout = open(args.transcript+".tmp", "w")
        text = txtin.readline()
        nrmtxt = norm(text)
        txtout.write(nrmtxt + "\n")
        txtout.close()
        
        # sclite   
        # first create a rec file
        transcr_wrd_starts = endpoints[0]
        transcr_wrd_ends = endpoints[1]
        transcr_wrd_txt = endpoints[2]
        
        with open(args.tempdir+"/basename.rec","w") as rec:
            for ix in range(0,len(transcr_wrd_txt)):
                rec.write(str(transcr_wrd_starts[ix])+" "+ str(transcr_wrd_ends[ix])+" "+transcr_wrd_txt[ix]+"\n")


    if args.outformat == "json":

        print(json.dumps(decoded_results))

    elif args.outformat == "textgrid":

        txg = textgrid.TextGrid() 
        txg.minTime = 0
        transcr_wrd_starts = endpoints[0]
        transcr_wrd_ends = endpoints[1]
        transcr_wrd_txt = endpoints[2]
        transcr_ph_starts = endpoints[3]
        transcr_ph_ends = endpoints[4]
        transcr_ph_txt = endpoints[5]

        offsets = decoded_results["output"][0]["offsets"]

        txg.endTime = offsets[len(offsets)-1]

        words_tier = textgrid.IntervalTier("words", 0, offsets[len(offsets)-1])
        phones_tier = textgrid.IntervalTier("phones", 0, offsets[len(offsets)-1])
        

        for jx in range(0, len(transcr_wrd_txt)-1):
            #print(transcr_wrd_txt[jx]+" "+str(transcr_wrd_starts[jx])+" "+str(transcr_wrd_ends[jx]))
            #if transcr_wrd_starts[jx] == transcr_wrd_ends[jx]: transcr_wrd_ends[jx] += 0.01
            words_tier.addInterval(textgrid.Interval(transcr_wrd_starts[jx], transcr_wrd_ends[jx], transcr_wrd_txt[jx]))

        for px in range(0, len(transcr_ph_txt)-1):
            #print(transcr_ph_txt[px]+" "+str(transcr_ph_starts[px])+" "+str(transcr_ph_ends[px]))
            #if transcr_ph_starts[jx] == transcr_ph_ends[jx]: transcr_ph_ends[jx] += 0.01
            #if transcr_ph_txt[px] == " ": transcr_ph_txt[px] = "SIL" 
            phones_tier.addInterval(textgrid.Interval(transcr_ph_starts[px], transcr_ph_ends[px], transcr_ph_txt[px]))
            #print(transcr[ix] + " " + str(offs[ix]))
        
        txg.append(words_tier)
        txg.append(phones_tier)

        txg.write(args.outputfile)

    # run praat2py
    pathtopraar2py = "/workspace/praat2py/praat2py/praat2py.py"
    os.system("python2.7 " + pathtopraar2py + " -t words -p /usr/bin/praat -o " + basedir + " " + args.audio_path + " " + args.outputfile)

    # cleanup
    os.path.exists(args.tempdir+"/"+basename+".pth.tar"):
        os.system('rm '+args.tempdir+"/"+basename+".pth.tar ")
    

