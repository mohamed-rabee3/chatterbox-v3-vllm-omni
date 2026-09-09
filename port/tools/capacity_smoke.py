"""Sequential functional renders, including the known early-EOS failure."""
import argparse,base64,json,time
from pathlib import Path
import requests,numpy as np,soundfile as sf

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
base='http://127.0.0.1:18091'
model=requests.get(base+'/v1/models',timeout=10).json()['data'][0]['id']
cases=[('early_eos_en','en','Good morning, you have reached the customer care line. What can I do for you?','en_spk1272',42),
 ('en_plain','en','Hello, this is the reference implementation speaking.','en_ex01',1234),
 ('ar_plain','ar','حياك الله، موعدك بكرة الساعة التاسعة صباحًا.','en_ex01',1234),
 ('mixed_ar_en','ar','حياك الله، حسابك على Netflix تم تجديده اليوم.','en_ex02',1234),
 ('en_long','en','There are two ways to do this. You can either keep the current plan and add the extra seats one at a time, which bills at the standard rate, or you can move to the annual plan, which includes ten seats and works out cheaper if you expect the team to grow. I can apply either one before the end of today’s call.','en_ex01',42)]
records=[]
for name,lang,text,voice,seed in cases:
 payload=dict(model=model,input=text,language=lang,ref_audio='data:audio/wav;base64,'+base64.b64encode(Path('/workspace/refvoices/'+voice+'.wav').read_bytes()).decode(),seed=seed,stream=True,stream_format='audio',response_format='pcm')
 start=time.perf_counter();chunks=[];first=None
 try:
  with requests.post(base+'/v1/audio/speech',json=payload,stream=True,timeout=300) as r:
   r.raise_for_status()
   for chunk in r.iter_content(None):
    if not chunk:continue
    if first is None:first=time.perf_counter()-start
    chunks.append(chunk)
  audio=np.frombuffer(b''.join(chunks),dtype='<i2').astype(np.float32)/32768
  sf.write(args.out/(name+'.wav'),audio,24000)
  rec=dict(name=name,text=text,language=lang,seed=seed,ttfa_s=first,audio_s=len(audio)/24000,finite=bool(np.isfinite(audio).all()),error=None)
 except Exception as e:rec=dict(name=name,text=text,error=str(e))
 records.append(rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
 (args.out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
raise SystemExit(any(r['error'] for r in records))
