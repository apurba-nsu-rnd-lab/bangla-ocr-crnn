import os

# Synthetic Train Set
# font_dir = "/media/blank/Hermes/Apurba/datasets/fonts/train_val/"
# wordlist_path = "/media/blank/Hermes/Apurba/datasets/dakshina/synthetic_wordlist_train.txt"
# output_dir = "/media/blank/Apollo/synthetic_words/train/"

# Synthetic Validation Set
# font_dir = "/media/blank/Hermes/Apurba/datasets/fonts/train_val/"
# wordlist_path = "/media/blank/Hermes/Apurba/datasets/dakshina/synthetic_wordlist_valid.txt"
# output_dir = "/media/blank/Apollo/synthetic_words/valid/"

# Synthetic Test set
# font_dir = "/media/blank/Hermes/Apurba/datasets/fonts_test/"
# wordlist_path = "/media/blank/Hermes/Apurba/datasets/dakshina/synthetic_wordlist_test.txt"
# output_dir = "/media/blank/Apollo/synthetic_ocr/test/"


# Synthetic Numbers set
font_dir = "/mnt/d/Apurba/OCR/datasets/fonts/train_val/"
wordlist_path = "/mnt/d/Apurba/OCR/datasets/dakshina/synthetic_wordlist_punct_num_pruned.txt"
output_dir = "/mnt/c/synthetic_wordlist_punct_num/train/"

# Synthetic fonts demo set
# font_dir = "/media/blank/Hermes/Apurba/datasets/fonts/demo/"
# wordlist_path = "/media/blank/Hermes/Apurba/datasets/dakshina/demo.txt"
# output_dir = "/media/blank/Apollo/synthetic_words/train/"


def main():
    
    # Define the font size
    font_sizes = [8, 10, 12, 18, 24, 32, 48, 64]

    margin = 14 # image margin

    # Image Sizes
    sizes = [size + margin for size in font_sizes]
    sizes_string = ""

    for size in sizes:
        sizes_string = sizes_string + str(size) + " "

    print(sizes_string)

    with open(wordlist_path, 'r') as wordfile:
        wordlist = [line.strip() for line in wordfile]
    

    word_count = len(wordlist)
    si = 1
    multiplicity = 1
    
    # vanilla command
    # os.system('python run.py -c 1 -i text.txt -l bn -w 1 -na 1')
    # python run.py -c 2 -i test.txt -l bn -w 1 -na 1


    for i in range(multiplicity):
        # on -f define the muliple sizes
        os.system("python3 run.py --output_dir {} -i {} -fd {} -c {} -f {} -si {} -na 2 -l bn -b 1 -t 4 -im \'L\'"
                    .format(output_dir, wordlist_path, font_dir, word_count, sizes_string, si))

        si += word_count

if __name__ == "__main__":
    main()
