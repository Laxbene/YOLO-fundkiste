
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os
import time
import random
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# --- 1. GLOBALE KONFIGURATION ---
HEUTE = datetime(2026, 3, 12).date()
DB_FILE = "fundstuecke_db.csv"
IMG_FOLDER = "images"
CONFIDENCE_THRESHOLD = 0.50 

if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

SPACE_WORDS = ["Asteroid", "Astronaut", "Apollo", "Atmosphäre", "Alien", "Galaxy", "Mars", "Rocket", "Star", "Universe"]

# --- 2. DATENBANK & LOGIK ---
def get_database():
    if os.path.exists(DB_FILE):
        try:
            return pd.read_csv(DB_FILE)
        except:
            return pd.DataFrame(columns=["ID", "Kategorie", "Funddatum", "Ablaufdatum", "Status", "Bild_Pfad"])
    return pd.DataFrame(columns=["ID", "Kategorie", "Funddatum", "Ablaufdatum", "Status", "Bild_Pfad"])

def save_database(df):
    df.to_csv(DB_FILE, index=False)

def delete_entry(entry_id):
    df = get_database()
    img_path = df.loc[df['ID'] == entry_id, 'Bild_Pfad'].values
    if len(img_path) > 0 and os.path.exists(str(img_path[0])):
        try: os.remove(str(img_path[0]))
        except: pass
    df = df[df['ID'] != entry_id]
    save_database(df)

@st.cache_resource
def load_yolo_model():
    model_path = 'best.pt' if os.path.exists('best.pt') else 'yolov8n.pt'
    try: return YOLO(model_path)
    except: return None

# --- 3. SEITEN-SETUP ---
st.set_page_config(page_title="Fundkiste Pro 2026", layout="wide")
model = load_yolo_model()

# Sidebar Navigation (Nur EINMAL definiert)
st.sidebar.title("🏢 Zentrale")
if 'page' not in st.session_state:
    st.session_state.page = "📸 Erfassen"

auswahl = st.sidebar.selectbox("Navigation", 
    ["📸 Erfassen", "📋 Kategorien-Galerie", "🔍 Suche", "🎮 Space Typing", "🚀 Doodle Jump"],
    index=["📸 Erfassen", "📋 Kategorien-Galerie", "🔍 Suche", "🎮 Space Typing", "🚀 Doodle Jump"].index(st.session_state.page))

# --- MODUS: ERFASSEN ---
if auswahl == "📸 Erfassen":
    st.header("📸 Fundstück-Analyse")
    
    # Check ob ein Bild aus der Galerie geschickt wurde
    target_image = None
    if 'pre_load_img' in st.session_state and st.session_state.pre_load_img:
        target_image = st.session_state.pre_load_img
        st.info("Bild aus Galerie geladen.")
        if st.button("Anderes Bild hochladen"):
            st.session_state.pre_load_img = None
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            target_image = Image.open(uploaded_file)

    if target_image and model:
        image = target_image.convert("RGB")
        st.image(image, caption="Scan-Vorgang...", width=400)
        
        # YOLO Prediction
        results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD)
        klasse = model.names[int(results[0].boxes[0].cls[0])] if len(results[0].boxes) > 0 else "Nicht erkannt"
        
        with st.form("save_form"):
            k_liste = list(model.names.values()) + ["Nicht erkannt"]
            final_klasse = st.selectbox("Kategorie bestätigen", k_liste, index=k_liste.index(klasse) if klasse in k_liste else 0)
            beschreibung = st.text_input("Zusatz-Info (Farbe, Zustand...)")
            if st.form_submit_button("In Datenbank speichern"):
                img_path = os.path.join(IMG_FOLDER, f"{int(time.time())}.jpg")
                image.save(img_path)
                df = get_database()
                neu = {"ID": int(time.time()), "Kategorie": final_klasse, "Funddatum": HEUTE, "Ablaufdatum": HEUTE+timedelta(days=30), "Status": beschreibung, "Bild_Pfad": img_path}
                save_database(pd.concat([df, pd.DataFrame([neu])], ignore_index=True))
                st.session_state.pre_load_img = None
                st.success("Gespeichert!")
                st.rerun()

# --- MODUS: KATEGORIEN-GALERIE ---
elif auswahl == "📋 Kategorien-Galerie":
    st.header("📋 Inventar-Galerie")
    df = get_database()
    if not df.empty:
        kategorien = sorted(df['Kategorie'].unique())
        for kat in kategorien:
            with st.expander(f"📁 {kat.upper()}", expanded=True):
                kat_items = df[df['Kategorie'] == kat]
                cols = st.columns(4)
                for i, (_, item) in enumerate(kat_items.iterrows()):
                    with cols[i % 4]:
                        if os.path.exists(str(item['Bild_Pfad'])):
                            st.image(item['Bild_Pfad'], use_container_width=True)
                            
                            # KNOPF: AN SCANNER SCHICKEN
                            if st.button("🔍 Re-Scan", key=f"rescan_{item['ID']}"):
                                st.session_state.pre_load_img = Image.open(item['Bild_Pfad'])
                                st.session_state.page = "📸 Erfassen"
                                st.rerun()
                                
                            if st.button("🗑️ Löschen", key=f"del_{item['ID']}"):
                                delete_entry(item['ID'])
                                st.rerun()
    else:
        st.info("Keine Daten vorhanden.")

# --- (Restliche Modi wie Suche, Spiele etc. hier einfügen) ---



# --- MODUS: SUCHE ---
elif auswahl == "🔍 Suche":
    st.header("🔍 Schnellsuche")
    query = st.text_input("Suchbegriff...")
    df = get_database()
    if query and not df.empty:
        res = df[df.apply(lambda r: query.lower() in r.astype(str).str.lower().values, axis=1)]
        st.dataframe(res, use_container_width=True)

# --- SPIELE SEKTION ---
elif auswahl == "🎮 Space Typing":
    st.header("☄️ Space Typer")
    if 'input_key' not in st.session_state: st.session_state.input_key = 0
    if 'game_active' not in st.session_state: st.session_state.game_active = False
    if not st.session_state.game_active:
        if st.button("Start"):
            st.session_state.game_active, st.session_state.lives, st.session_state.score, st.session_state.current_word, st.session_state.start_time = True, 3, 0, random.choice(SPACE_WORDS), time.time()
            st.rerun()
    else:
        rest = max(0.0, 7.0 - (time.time() - st.session_state.start_time))
        st.write(f"### Wort: :orange[{st.session_state.current_word}] | ❤️ {st.session_state.lives} | ⭐ {st.session_state.score}")
        st.progress(rest / 7.0)
        fid = f"typer_{st.session_state.input_key}"
        ui = st.text_input("Tippen:", key=fid).strip()
        components.html(f"<script>window.parent.document.querySelector('input[id*=\"{fid}\"]').focus();</script>", height=0)
        if ui.lower() == st.session_state.current_word.lower():
            st.session_state.score += 10; st.session_state.current_word = random.choice(SPACE_WORDS); st.session_state.start_time = time.time(); st.session_state.input_key += 1; st.rerun()
        if rest <= 0:
            st.session_state.lives -= 1; st.session_state.start_time = time.time(); st.session_state.input_key += 1
            if st.session_state.lives <= 0: st.session_state.game_active = False
            st.rerun()
        time.sleep(0.1); st.rerun()

elif auswahl == "⚡ Reaktionstest":
    st.header("⚡ Reaktionstest")
    if 'rxn_state' not in st.session_state: st.session_state.rxn_state = "idle"
    if st.session_state.rxn_state == "idle":
        if st.button("Start"): st.session_state.rxn_state = "waiting"; st.session_state.wait_until = time.time() + random.uniform(2, 5); st.rerun()
    elif st.session_state.rxn_state == "waiting":
        st.error("### WARTEN..."); (time.sleep(0.05) or st.rerun()) if time.time() < st.session_state.wait_until else (setattr(st.session_state, 'rxn_state', 'go') or setattr(st.session_state, 'go_start', time.time()) or st.rerun())
    elif st.session_state.rxn_state == "go":
        if st.button("KLICK!"): st.session_state.last_res = (time.time() - st.session_state.go_start)*1000; st.session_state.rxn_state = "result"; st.rerun()
    elif st.session_state.rxn_state == "result":
        st.write(f"## {st.session_state.last_res:.0f} ms"); (st.button("Nochmal") and setattr(st.session_state, 'rxn_state', 'idle') or st.rerun())

elif auswahl == "🎯 Aim-Trainer":
    st.header("🎯 Aim-Trainer")
    if 'aim_hits' not in st.session_state: st.session_state.aim_hits = 0
    if st.session_state.aim_hits == 0:
        if st.button("Start"): st.session_state.aim_hits = 1; st.session_state.aim_start = time.time(); st.rerun()
    elif st.session_state.aim_hits <= 10:
        c = st.columns(10); (c[random.randint(0, 9)].button("🎯", key=f"aim_{st.session_state.aim_hits}") and setattr(st.session_state, 'aim_hits', st.session_state.aim_hits + 1) or st.rerun())
    else:
        st.write(f"## Zeit: {time.time()-st.session_state.aim_start:.2f}s"); (st.button("Reset") and setattr(st.session_state, 'aim_hits', 0) or st.rerun())
        
# --- MODUS: DOODLE JUMP (FAIR-PLAY UPDATE) ---
elif auswahl == "🚀 Doodle Jump":
    st.header("🚀 Space Jumper - Pro Edition")
    st.info("Steuerung: Pfeiltasten LINKS/RECHTS. Jede Lücke ist springbar!")
    
    doodle_html = """
    <canvas id="gameCanvas" width="400" height="600" style="border:3px solid #444; display:block; margin:auto; background:#fcf5f9;"></canvas>
    <script>
        const canvas = document.getElementById('gameCanvas'), ctx = canvas.getContext('2d');
        let player = { x: 180, y: 450, w: 35, h: 45, vy: 0, vx: 0 };
        let platforms = [], score = 0, keys = {};
        const gravity = 0.25, jumpPower = -9.5; // Leicht erhöhte Sprungkraft für Sicherheit

        function createPlatform(y, isBase=false) {
            let type = 'normal';
            if(!isBase) {
                let r = Math.random();
                if(r > 0.90) type = 'boost'; 
                else if(r > 0.75) type = 'broken';
            } else type = 'base';
            
            // x-Position so wählen, dass sie nicht zu weit am Rand klebt
            let x = isBase ? 100 : Math.random() * 320; 
            return { x: x, y: y, w: isBase ? 200 : 70, h: 12, type: type };
        }

        function init() {
            score = 0; player.x = 180; player.y = 450; player.vy = 0;
            platforms = [];
            // Bodenplatte
            platforms.push(createPlatform(550, true));
            // Plattformen mit festem Maximalabstand (max 85 Pixel vertikal)
            // Das garantiert, dass man mit jumpPower (-9.5) immer hochkommt
            for(let i=0; i<7; i++) {
                platforms.push(createPlatform(550 - (i + 1) * 85));
            }
        }

        function update() {
            player.vy += gravity;
            player.y += player.vy;
            
            if(keys['ArrowLeft']) player.vx = -5;
            else if(keys['ArrowRight']) player.vx = 5;
            else player.vx *= 0.8; // Sanftes Abbremsen
            
            player.x += player.vx;

            // Wrap-around
            if(player.x < -30) player.x = canvas.width;
            if(player.x > canvas.width) player.x = -30;

            // Kamera-Follow
            if(player.y < 250) {
                let delta = 250 - player.y;
                player.y = 250;
                platforms.forEach(p => {
                    p.y += delta;
                    if(p.y > 600) {
                        score++;
                        // Neue Plattform immer oben im Bereich 0-20px generieren
                        Object.assign(p, createPlatform(p.y - 600));
                    }
                });
            }

            // Kollision (nur beim Runterfallen)
            if(player.vy > 0) {
                platforms.forEach(p => {
                    if(player.x + player.w > p.x && player.x < p.x + p.w &&
                       player.y + player.h > p.y && player.y + player.h < p.y + 15) {
                        
                        if(p.type === 'boost') {
                            player.vy = jumpPower * 1.8;
                        } else if(p.type === 'broken') {
                            player.vy = jumpPower;
                            p.y = 1000; // Block "zerstört"
                        } else {
                            player.vy = jumpPower;
                        }
                    }
                });
            }

            if(player.y > 600) init(); // Game Over
        }

        function draw() {
            ctx.clearRect(0,0,canvas.width,canvas.height);
            
            // Astronaut zeichnen
            ctx.fillStyle = '#ff4b4b'; ctx.fillRect(player.x, player.y, player.w, player.h);
            ctx.fillStyle = '#88ccff'; ctx.fillRect(player.x+5, player.y+8, player.w-10, 15);

            // Blöcke zeichnen
            platforms.forEach(p => {
                if(p.type==='boost') ctx.fillStyle='#f1c40f';
                else if(p.type==='broken') ctx.fillStyle='#ffffff';
                else ctx.fillStyle='#2ecc71';
                ctx.fillRect(p.x, p.y, p.w, p.h);
            });

            ctx.fillStyle = '#03396c'; ctx.font = 'bold 20px Courier';
            ctx.fillText("SCORE: " + score, 20, 40);
        }

        window.onkeydown = e => { keys[e.key] = true; if(e.key.includes("Arrow")) e.preventDefault(); };
        window.onkeyup = e => keys[e.key] = false;

        init();
        function main() { update(); draw(); requestAnimationFrame(main); }
        main();
    </script>
    """
    components.html(doodle_html, height=650)
