import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# --- KONFIGURATION ---
HEUTE = datetime(2026, 3, 12).date()
DB_FILE = "fundstuecke_db.csv"
IMG_FOLDER = "images"
CONFIDENCE_THRESHOLD = 0.50 

# Verzeichnisse erstellen
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

# --- KI MODELL LADEN (Cache verhindert Neu-Laden bei jedem Klick) ---
@st.cache_resource
def load_yolo_model():
    try:
        # Versuche das Modell zu laden. 
        # Falls 'best.pt' nicht da ist, nimmt er das Standard 'yolov8n.pt'
        model_path = 'best.pt' if os.path.exists('best.pt') else 'yolov8n.pt'
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden: {e}")
        return None

# --- DATENBANK-FUNKTIONEN ---
def get_database():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(columns=["ID", "Kategorie", "Funddatum", "Ablaufdatum", "Status", "Bild_Pfad"])

def save_database(df):
    df.to_csv(DB_FILE, index=False)

def delete_entry(entry_id):
    df = get_database()
    # Bild löschen
    path_row = df.loc[df['ID'] == entry_id, 'Bild_Pfad']
    if not path_row.empty and os.path.exists(str(path_row.values[0])):
        os.remove(str(path_row.values[0]))
    # Zeile entfernen
    df = df[df['ID'] != entry_id]
    save_database(df)

# --- UI SETUP ---
st.set_page_config(page_title="Fundkiste AI 2026", layout="wide")
model = load_yolo_model()

st.sidebar.title("🏢 Zentrale")
auswahl = st.sidebar.selectbox("Navigation", 
    ["📸 Erfassen", "📊 Datenbank", "🔍 Suche", "🚀 Doodle Jump"])

# --- KONFIGURATION ---
HEUTE = datetime(2026, 3, 12).date()
DB_FILE = "fundstuecke_db.csv"
IMG_FOLDER = "images"
CONFIDENCE_THRESHOLD = 0.50  # YOLO ist oft präziser, 0.50 ist ein guter Startwert

if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

SPACE_WORDS = ["Asteroid", "Astronaut", "Apollo", "Atmosphäre", "Antimaterie", "Alien", "Aurora", "Blackhole", "Comet", "Cosmos", "Darkmatter", "Deepspace", "Eclipse", "Exoplanet", "Galaxy", "Gravity", "Hubble", "Interstellar", "Jupiter", "Kepler", "Mars", "Meteor", "Milkyway", "Moon", "Nebula", "Neptune", "Orbit", "Orion", "Planet", "Pluto", "Rocket", "Rover", "Saturn", "Shuttle", "Star", "Supernova", "Telescope", "Universe", "Uranus", "Venus", "Voyager", "Warp", "Zenith"]

# --- DATENBANK-FUNKTIONEN ---
def get_database():
    if os.path.exists(DB_FILE):
        try:
            return pd.read_csv(DB_FILE)
        except Exception as e:
            st.error(f"Fehler beim Laden der DB: {e}")
            return pd.DataFrame(columns=["ID", "Kategorie", "Funddatum", "Ablaufdatum", "Status", "Bild_Pfad"])
    return pd.DataFrame(columns=["ID", "Kategorie", "Funddatum", "Ablaufdatum", "Status", "Bild_Pfad"])

def save_database(df):
    try:
        df.to_csv(DB_FILE, index=False)
    except Exception as e:
        st.error(f"Speichern fehlgeschlagen: {e}")

def delete_entry(entry_id):
    df = get_database()
    img_to_delete = df.loc[df['ID'] == entry_id, 'Bild_Pfad'].values
    if len(img_to_delete) > 0 and os.path.exists(str(img_to_delete[0])):
        try: os.remove(str(img_to_delete[0]))
        except: pass
    df = df[df['ID'] != entry_id]
    save_database(df)

# --- YOLO KI MODELL LADEN ---
@st.cache_resource
def load_yolo_model():
    # Hier den Pfad zu deiner YOLO-Datei (.pt) angeben
    # Wenn du ein exportiertes Keras-Modell hast, nutze 'best.pt' oder 'yolov8n.pt'
    try:
        model = YOLO('best.pt') # Ersetze 'best.pt' durch deinen Dateinamen
        return model
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden: {e}")
        return None

# --- UI SETUP ---
st.set_page_config(page_title="Fundkiste AI 2026", layout="wide")
model = load_yolo_model()

st.sidebar.title("🏢 Zentrale")
auswahl = st.sidebar.selectbox("Navigation", 
    ["📸 Erfassen", "📊 Datenbank", "📋 Kategorien-Galerie", "🔍 Suche", "🎮 Space Typing", "🚀 Doodle Jump"])

# --- MODUS: ERFASSEN ---
if auswahl == "📸 Erfassen":
    st.header("📸 Neues Fundstück mit YOLO erfassen")
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Vorschau", width=400)
        
        # YOLO Prediction
        results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD)
        
        # Ergebnisse verarbeiten
        if len(results[0].boxes) > 0:
            # Wir nehmen das Objekt mit der höchsten Confidence
            box = results[0].boxes[0]
            class_id = int(box.cls[0])
            klasse = model.names[class_id]
            confidence = float(box.conf[0])
            
            st.success(f"✅ Objekt erkannt: **{klasse}** ({confidence:.1%})")
            
            # Box auf dem Bild anzeigen (optionaler Visualisierungs-Schritt)
            res_plotted = results[0].plot() # Gibt ein BGR Array zurück
            # st.image(res_plotted[:, :, ::-1], caption="KI Analyse", width=400)
        else:
            st.warning("⚠️ Kein bekanntes Objekt erkannt.")
            klasse = "Nicht erkannt"

        with st.form("save_form"):
            # Kategorien aus dem Modell laden
            k_liste = list(model.names.values())
            if "Nicht erkannt" not in k_liste: k_liste.append("Nicht erkannt")
            
            final_klasse = st.selectbox("Kategorie bestätigen", k_liste, index=k_liste.index(klasse) if klasse in k_liste else 0)
            beschreibung = st.text_input("Zusatz-Info (Farbe, Zustand...)")
            submit = st.form_submit_button("In Datenbank speichern")
            
            if submit:
                img_path = os.path.join(IMG_FOLDER, f"{int(time.time())}.jpg")
                image.save(img_path)
                df = get_database()
                neu = {
                    "ID": int(time.time()), 
                    "Kategorie": final_klasse, 
                    "Funddatum": HEUTE, 
                    "Ablaufdatum": HEUTE+timedelta(days=30), 
                    "Status": beschreibung, 
                    "Bild_Pfad": img_path
                }
                save_database(pd.concat([df, pd.DataFrame([neu])], ignore_index=True))
                st.success("Erfolgreich gespeichert!")

# --- MODUS: DATENBANK ---
elif auswahl == "📊 Datenbank":
    st.header("📊 Alle Fundstücke")
    df = get_database()
    if not df.empty:
        for _, row in df.iterrows():
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
            with c1:
                path = str(row['Bild_Pfad'])
                if os.path.exists(path): st.image(path, width=120)
                else: st.write("🖼️")
            with c2: st.write(f"**{row['Kategorie']}**\n\n{row['Status']}")
            with c3: st.write(f"📅 {row['Funddatum']}\n\n⏰ Ablauf: {row['Ablaufdatum']}")
            with c4: 
                if st.button("✅ Abgeholt", key=f"del_{row['ID']}"):
                    delete_entry(row['ID']); st.rerun()
            st.divider()

# --- SPIELE & REST (Doodle Jump bleibt gleich wie in deinem Code) ---
elif auswahl == "🚀 Doodle Jump":
    # Hier fügst du deinen Doodle Jump HTML/JS Code wieder ein
    st.header("🚀 Space Jumper")
    # ... (Code aus deinem Original einfügen)
