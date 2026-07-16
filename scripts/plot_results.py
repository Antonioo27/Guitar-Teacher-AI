import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_f1_comparison(json_path, out_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    # Raggruppa i risultati per file
    file_metrics = {}
    for r in results:
        fname = r['file']
        model = r['model']
        f1 = r['f1']
        
        if fname not in file_metrics:
            file_metrics[fname] = {}
        file_metrics[fname][model] = f1
        
    files = list(file_metrics.keys())
    files.sort()  # Ordina i file in ordine alfabetico
    
    # Identifica tutti i modelli presenti nei risultati (dovrebbero essere TabCNN e CRNN)
    models = set()
    for m in file_metrics.values():
        models.update(m.keys())
    models = sorted(list(models))
    
    if len(models) < 2:
        print(f"Attenzione: Trovato solo {len(models)} modello(i) nel file JSON.")
        print("Per un confronto diretto efficace, il file dovrebbe contenere i risultati di entrambi i modelli.")
        if len(models) == 0:
            return
            
    # Prepara i dati per l'istogramma
    x = np.arange(len(files))
    width = 0.35  # Larghezza delle barre
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Colori con tonalità leggermente diverse per distinguere bene i modelli
    # Usa due tonalità di rosso (una scura, una chiara)
    colors = ['#d62728', '#ff9896']
    
    for i, model in enumerate(models):
        scores = [file_metrics[f].get(model, 0.0) for f in files]
        # Calcola l'offset per affiancare le barre
        offset = width * i - (width * (len(models) - 1) / 2)
        ax.bar(x + offset, scores, width, label=model, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Confronto Diretto F1 Score: TabCNN vs CRNN', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    # Ruota le etichette dell'asse X per evitare sovrapposizioni
    ax.set_xticklabels(files, rotation=45, ha='right', fontsize=9)
    ax.legend(title='Modelli')
    
    # Aggiungi una griglia orizzontale per facilitare la lettura
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Margini per non tagliare il testo in basso
    fig.tight_layout()
    
    # Salva l'immagine
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"📊 Grafico salvato con successo in: {out_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Genera un istogramma per confrontare l'F1 score dei modelli.")
    parser.add_argument("--input", type=str, default="scripts/out/eval_results.json",
                        help="Percorso al file JSON con i risultati (default: scripts/out/eval_results.json)")
    parser.add_argument("--output", type=str, default="scripts/out/comparison_plot.png",
                        help="Percorso in cui salvare l'immagine generata (default: scripts/out/comparison_plot.png)")
                        
    args = parser.parse_args()
    
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"❌ Errore: il file di input '{in_path}' non esiste.")
        print("💡 Assicurati di eseguire prima lo script evaluate.py per generare i risultati JSON.")
        return
        
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_f1_comparison(in_path, out_path)

if __name__ == "__main__":
    main()
