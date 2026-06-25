"""
main.py - Run the full sentiment analysis pipeline from command line
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))


def run_pipeline(input_csv=None, output_visuals=True):
    from data_cleaning import load_and_clean
    from sentiment_model import analyze_dataframe, evaluate_model, save_model_info
    from visualization import generate_all_visuals

    raw_dir      = os.path.join(BASE_DIR, 'data', 'raw')
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    visuals_dir  = os.path.join(BASE_DIR, 'visuals')
    models_dir   = os.path.join(BASE_DIR, 'models')

    for d in [raw_dir, processed_dir, visuals_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    if input_csv is None:
        input_csv = os.path.join(raw_dir, 'social_media_data.csv')

    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        sys.exit(1)

    print(f"\n[1/4] Loading and cleaning data from: {input_csv}")
    df = load_and_clean(input_csv)
    print(f"      → {len(df)} records loaded")

    cleaned_path = os.path.join(processed_dir, 'cleaned_data.csv')
    df.to_csv(cleaned_path, index=False)
    print(f"      → Saved to: {cleaned_path}")

    print("\n[2/4] Running sentiment analysis (VADER + TextBlob)…")
    df = analyze_dataframe(df)

    metrics = evaluate_model(df)
    if metrics:
        print(f"      → Accuracy: {metrics.get('accuracy', 'N/A')}")
    
    df.to_csv(cleaned_path, index=False)
    save_model_info(metrics, models_dir)

    if output_visuals:
        print("\n[3/4] Generating visualizations…")
        paths = generate_all_visuals(df, visuals_dir)
        for key, path in paths.items():
            print(f"      → {key}: {os.path.basename(path)}")
    
    print("\n[4/4] Pipeline complete!")
    print(f"      Total posts analyzed : {len(df)}")
    
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    if sentiment_col in df.columns:
        for sent, count in df[sentiment_col].value_counts().items():
            print(f"      {sent:10s}: {count}")
    
    print(f"\n  Run the web app with:  python app.py")
    return df


def main():
    parser = argparse.ArgumentParser(description='Social Media Sentiment Analysis Pipeline')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input CSV file')
    parser.add_argument('--no-visuals', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--serve', action='store_true',
                        help='Start web server after pipeline')
    args = parser.parse_args()

    df = run_pipeline(
        input_csv=args.input,
        output_visuals=not args.no_visuals
    )

    if args.serve:
        import subprocess
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'app.py')])


if __name__ == '__main__':
    main()
