import requests, json, re, html, time, pathlib
S=requests.Session(); S.headers["User-Agent"]="FaceAIStudio/1.0 (melvindave@gmail.com)"
API="https://commons.wikimedia.org/w/api.php"
PICKS = {
 "people/reid_wiseman": [("Reid Wiseman","Jsc2023e0016434 alt"),("Reid Wiseman","2023 Gregory Reid Wiseman"),("Reid Wiseman","Reid-wiseman 51732552119"),("Reid Wiseman","Reid Wiseman in a NASA news conference")],
 "people/victor_glover": [("Victor Glover","Victor J. Glover official portrait"),("Victor Glover","Jsc2023e0016433 alt"),("Victor Glover","Victor Glover after Crew-1 splashdown"),("Victor Glover","Victor Glover poses in JEM")],
 "people/christina_koch": [("Christina Koch","Christina M. Hammock portrait"),("Christina Koch","Jsc2023e0016435 alt"),("Christina Koch","Christina Koch in a NASA news conference"),("Christina Koch","ISS-60 Christina Koch with a green bubble")],
 "people/jeremy_hansen": [("Jeremy Hansen astronaut","Jeremy Hansen official portrait"),("Jeremy Hansen astronaut","Jsc2023e0016436 alt2"),("Jeremy Hansen astronaut","NEEMO 19 Jeremy Hansen"),("Jeremy Hansen astronaut","Jeremy Hansen Looks Back Home")],
 "groups": [("Artemis II crew","Artemis 2 Crew Portrait"),("Artemis II crew","Artemis II Crew at the White House"),("Jeremy Hansen astronaut","Jsc2026e022297"),("Artemis II crew","Artemis II Crew geology training in Iceland"),("Jeremy Hansen astronaut","Astronaut class group 20"),("Barack Obama official portrait","Barack Obama family portrait 2011"),("astronaut crew portrait","Expedition 49 crew portrait")],
}
def clean(s): return html.unescape(re.sub("<[^>]+>","",s or "")).strip()
credits=[]
for folder, items in PICKS.items():
    d=pathlib.Path("photos")/folder; d.mkdir(parents=True, exist_ok=True)
    for q,prefix in items:
        r=S.get(API,params=dict(action="query",format="json",generator="search",gsrsearch=q+" filetype:bitmap",gsrnamespace=6,gsrlimit=30,prop="imageinfo",iiprop="url|extmetadata",iiurlwidth=1600)).json()
        pages=[p for p in r["query"]["pages"].values() if p["title"][5:].startswith(prefix)]
        if not pages: print("MISSING",prefix); continue
        p=pages[0]; ii=p["imageinfo"][0]; m=ii["extmetadata"]
        slug=re.sub(r"[^a-z0-9]+","_",prefix.lower()).strip("_")[:50]
        name=f"{slug}.jpg"; out=d/name
        for a in range(4):
            resp=S.get(ii["thumburl"]); 
            if resp.ok: break
            time.sleep(3*(a+1))
        resp.raise_for_status()
        if "webp" in resp.headers.get("content-type",""):
            from PIL import Image; import io; Image.open(io.BytesIO(resp.content)).convert("RGB").save(out,quality=92)
        else: out.write_bytes(resp.content)
        credits.append(dict(file=str(out), title=p["title"], source=ii["descriptionurl"], author=clean(m.get("Artist",{}).get("value")), credit=clean(m.get("Credit",{}).get("value"))[:200], license=m.get("LicenseShortName",{}).get("value")))
        print("ok",out, credits[-1]["license"], len(resp.content)//1024,"KB")
        time.sleep(1)
json.dump(credits, open("photos/CREDITS.json","w"), indent=2)
