from transformers import pipeline
import logging, os, csv

fill_mask = pipeline('fill-mask', model='distilbert-base-uncased')
mask = fill_mask.tokenizer.mask_token

def get_typo_locations(fh):
    tsv_f = csv.reader(fh, delimiter='\t')
    for line in tsv_f:
        yield (
            # line[0] contains the comma separated indices of typo words
            [int(i) for i in line[0].split(',')],
            # line[1] contains the space separated tokens of the sentence
            line[1].split()
        )
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def select_correction(typo, predict):
    filtered_predict = [p for p in predict if "##" not in p["token_str"]]
    
    if not filtered_predict:
        filtered_predict = predict
    
    # Combine Levenshtein distance and model confidence to select the best correction
    def combined_score(prediction):
        levenshtein_score = levenshtein_distance(typo, prediction["token_str"])
        confidence_score = prediction.get("score", 0)  
        return levenshtein_score - confidence_score  
    
    closest_prediction = min(filtered_predict, key=combined_score)
    return closest_prediction["token_str"]


def spellchk(fh):
     for (locations, sent) in get_typo_locations(fh):
        spellchk_sent = sent.copy()  # Create a copy of the sentence tokens
        for i in locations:
            # Formulate the sentence with a masked token for the current typo
            masked_sentence = " ".join(sent[j] if j != i else mask for j in range(len(sent)))
            # Generate predictions for the masked token
            predict = fill_mask(masked_sentence, top_k=180)

            correction = select_correction(sent[i], predict)
            spellchk_sent[i] = correction
        
        yield (locations, spellchk_sent)


   
if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", 
                            dest="input", 
                            default=os.path.join('data', 'input', 'dev.tsv'), 
                            help="file to segment")
    argparser.add_argument("-l", "--logfile", 
                            dest="logfile", 
                            default=None, 
                            help="log file for debugging")
    opts = argparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    with open(opts.input) as f:
        for (locations, spellchk_sent) in spellchk(f):
            print("{locs}\t{sent}".format(
                locs=",".join([str(i) for i in locations]),
                sent=" ".join(spellchk_sent)
            ))
