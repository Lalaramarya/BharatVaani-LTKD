import os
import subprocess
from tqdm import tqdm
import re
import argparse

# Define the language code dictionary

language_codes = {
    "English": "en", "Bodo": "as", "Chattisgarhi": "hi", "Dogri": "hi",
    "Garo": "bd", "Galo": "bd", "Jaintia": "bd", "Kashmiri": "ur",
    "Khasi": "bd", "Kokborok": "bn", "Konkani": "hi", "Ladakhi": "bd",
    "Lepcha": "bd", "Maithili": "hi", "Mizo": "bd", "Nepali": "hi",
    "Purgi": "ur", "Sanskrit": "hi", "Santhali": "bn",
    "Sargujia": "hi", "Sikkimese": "bd", "Sindhi": "hi",
    "Assamese": "as", "Bengali": "bn", "Gujarathi": "gu",
    "Hindi": "hi", "Kannada": "kn", "Malayalam": "ml", "Manipuri": "mni",
    "Marathi": "mr", "Odia": "or", "Punjabi": "pa", "Tamil": "ta",
    "Telugu": "te", "Urdu": "ur"
}

# Function to extract episode number from folder name
def extract_episode_number(folder_name):
    match = re.search(r'MKB_(\d+)', folder_name)
    if match:
        return int(match.group(1))  # Extract the number and return it as an integer
    return float('inf')  # If no match, return a very high number to sort at the end

# Function to process the alignment and segmentation for each subfolder
def process_alignment(language, episode_folder, base_dir, uroman_path, output_dir):
    # Get the full path to the text and audio files
    text_file = os.path.join(base_dir, language, episode_folder, f"{episode_folder}_split_sentences.txt")
    #text_file = os.path.join(base_dir, language, episode_folder, f"{episode_folder}_en_aligned.txt")
    audio_file = os.path.join(base_dir, language, episode_folder, f"{episode_folder}.wav")

    # Skip if the text or audio file is missing
    if not os.path.exists(text_file) or not os.path.exists(audio_file):
        print(f"⚠️ Skipping {episode_folder} - missing text or audio file.")
        return

    # Construct the output folder path for this episode
    output_folder = os.path.join(output_dir, language, episode_folder,episode_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Construct the command with the required parameters
    command = [
        "python", "./MMS-forced-align/align_and_segment.py",
        "--audio_filepath", audio_file,
        "--text_filepath", text_file,
        "--lang", language_codes[language],  # Use language code from dictionary
        "--uroman_path", uroman_path,
        "--outdir", output_folder
    ]

    # Run the command
    try:
        print(f"Processing {language} - {episode_folder}...")
        subprocess.run(command, check=True)
        print(f"✅ Successfully processed {language} - {episode_folder}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {language} - {episode_folder}: {e}")

# Main function to iterate through all the languages and their episode subfolders
def main(base_dir, uroman_path, output_dir):
    # Iterate through each language folder
    for language in tqdm(language_codes,desc="Processing across languages"):
        language_path = os.path.join(base_dir, language)

        # Ensure the language folder exists
        if not os.path.exists(language_path):
            print(f"Warning: Language folder for {language} not found!")
            continue

        # List all the episode folders (subfolders) and sort them by episode number using regex
        episode_folders = sorted(
            os.listdir(language_path),
            key=lambda x: extract_episode_number(x),reverse=True
        )
        
        if language in ["English"]:
            start_idx = episode_folders.index("MKB_59_November_2019")
            episode_folders = episode_folders[start_idx:]

        # Use tqdm to show the progress bar while processing episode folders
        for episode_folder in tqdm(episode_folders, desc=f"Processing {language}"):
            if os.path.isdir(os.path.join(language_path, episode_folder)):  # Ensure it's a folder
                process_alignment(language, episode_folder, base_dir, uroman_path, output_dir)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process multilingual forced alignment and segmentation.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for language folders.")
    parser.add_argument('--uroman_path', type=str, required=True, help="Path to the uroman binary directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output results.")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.base_dir, args.uroman_path, args.output_dir)

