import asyncio, json, time, sys, websockets
async def go():
    try:
        async with websockets.connect('wss://chat.mikeayles.com/', open_timeout=10, max_size=2**20) as ws:
            await asyncio.wait_for(ws.recv(), 10)
            t0=time.time(); await ws.send(json.dumps({'type':'enter','prompt':'once upon a time','seq':1,'blitted':False}))
            st=''
            while time.time()-t0<15:
                m=json.loads(await asyncio.wait_for(ws.recv(),15))
                if m.get('type')=='stream': st+=m.get('text','')
                if m.get('type') in ('authoritative','stream_end') and m.get('prompt')=='once upon a time':
                    c=(m.get('completion') or st); rtt=(time.time()-t0)*1000
                    caps=sum(x.isupper() for x in c); frac=sum(x.isalpha() or x==' ' for x in c)/max(len(c),1)
                    flag='EMPTY' if not c.strip() else ('GARBAGE' if frac<0.7 else ('LEGIT?' if caps>2 else 'kevin?'))
                    print('%s  RTT %4.0fms  %-8s :: %r'%(time.strftime('%H:%M:%S'),rtt,flag,c[:52])); return
            print('%s  NO-RESPONSE'%time.strftime('%H:%M:%S'))
    except Exception as e:
        print('%s  ERR %s'%(time.strftime('%H:%M:%S'), repr(e)[:50]))
asyncio.run(go())
