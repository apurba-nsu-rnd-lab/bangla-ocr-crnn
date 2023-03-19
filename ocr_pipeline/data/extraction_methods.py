"""
This file contains three of our unique grapheme extraction methods
Naive -> Break down all unicodes
VDS -> Vowel Diacritics Seperation
ADS -> All Diacritics Seperation
"""

def normalize_word(word):

    if 'ো' in word: word = word.replace('ো', 'ো')
    
    if 'ৗ' in word:    
        if 'ৌ' in word: word = word.replace('ৌ', 'ৌ') 
        else: word = word.replace('ৗ', 'ী') # 'ৗ' without 'ে' is replaced by 'ী'
    
    if '়' in word:
        if 'ব়' in word: word = word.replace('ব়', 'র')
        if 'য়' in word: word = word.replace('য়', 'য়')
        if 'ড়' in word: word = word.replace('ড়', 'ড়')
        if 'ঢ়' in word: word = word.replace('ঢ়', 'ঢ়')
        if '়' in word: word = word.replace('়', '') # discard any other '়' without 'ব'/'য'/'ড'/'ঢ'
        
    # visually similar '৷' (Bengali Currency Numerator Four) is replaced by '।' (Devanagari Danda)
    if '৷' in word: word = word.replace('৷', '।')
    
    return word



################################# Naive character representation #################################
def naive_grapheme_extraction(word):
    
    return list(word)



################################# Vowel Diacritics Seperation #################################
def vds_grapheme_extraction(word):
    
    consonants = ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 
                  'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়']
    
    forms_cluster = {'ক': ['ক', 'ট', 'ত', 'ন', 'ব', 'ম', 'র', 'ল', 'ষ', 'স'],
                     'খ': ['র'],
                     'গ': ['গ', 'ধ', 'ন', 'ব', 'ম', 'র', 'ল'],
                     'ঘ': ['ন', 'র'],
                     'ঙ': ['ক', 'খ', 'গ', 'ঘ', 'ম', 'র'],
                     'চ': ['চ', 'ছ', 'ঞ', 'র'],
                     'ছ': ['র'],
                     'জ': ['জ', 'ঝ', 'ঞ', 'ব', 'র'],
                     'ঝ': ['র'],
                     'ঞ': ['চ', 'ছ', 'জ', 'ঝ', 'র'],
                     'ট': ['ট', 'ব', 'র'],
                     'ঠ': ['র'],
                     'ড': ['ড', 'র'],
                     'ঢ': ['র'],
                     'ণ': ['ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ব', 'ম', 'র'],
                     'ত': ['ত', 'থ', 'ন', 'ব', 'ম', 'র'],
                     'থ': ['ব', 'র'],
                     'দ': ['গ', 'ঘ', 'দ', 'ধ', 'ব', 'ভ', 'ম', 'র'],
                     'ধ': ['ন', 'ব', 'র'],
                     'ন': ['জ', 'ট', 'ঠ', 'ড', 'ত', 'থ', 'দ', 'ধ', 'ন', 'ব', 'ম', 'র', 'স'],
                     'প': ['ট', 'ত', 'ন', 'প', 'ল', 'র', 'স'],
                     'ফ': ['ট', 'র', 'ল'],
                     'ব': ['জ', 'দ', 'ধ', 'ব', 'ভ', 'র', 'ল'],
                     'ভ': ['র'],
                     'ম': ['ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'র', 'ল'],
                     'য': ['র'],
                     'র': ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ',
                           'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়'],
                     'ল': ['ক', 'গ', 'ট', 'ড', 'প', 'ফ', 'ব', 'ম', 'র', 'ল', 'স'],
                     'শ': ['চ', 'ছ', 'ত', 'ন', 'ব', 'ম', 'র', 'ল'],
                     'ষ': ['ক', 'ট', 'ঠ', 'ণ', 'প', 'ফ', 'ব', 'ম', 'র'],
                     'স': ['ক', 'খ', 'ট', 'ত', 'থ', 'ন', 'প', 'ফ', 'ব', 'ম', 'র', 'ল'],
                     'হ': ['ণ', 'ন', 'ব', 'ম', 'র', 'ল'],
                     'ড়': ['গ', 'র'],
                     'ঢ়': ['র'],
                     'য়': ['র']}
    
    
    forms_tripple_cluster = {'ক্ট': ['র'], 'ক্ত': ['র'], 'ক্ষ': ['ণ', 'ম', 'র'], 'ঙ্ক': ['ষ', 'র'], 'চ্ছ': ['ব', 'র'], 'জ্জ': ['ব'],
                             'ণ্ড': ['র'], 'ত্ত': ['ব'], 'দ্দ': ['ব'], 'দ্ধ': ['ব'], 'দ্ভ': ['র'], 'ন্ট': ['র'], 'ন্ড': ['র'], 'ন্ত': ['ব', 'র'],
                             'ন্দ': ['ব', 'র'], 'ন্ধ': ['র'], 'ম্প': ['র', 'ল'], 'ম্ভ': ['র'],
                             'ষ্ক': ['র'], 'স্ক': ['র'], 'ষ্ট': ['র'], 'স্ট': ['র'], 'স্ত': ['ব', 'র'], 'ষ্প': ['র'], 'স্প': ['র', 'ল'],
                             # refs
                             'র্ক': ['র', 'ট', 'ত', 'ল', 'ষ', 'স'], 'র্খ': ['র'], 'র্গ': ['ব', 'ল', 'র'], 'র্ঘ': ['র'], 'র্ঙ': ['ক', 'গ', 'র'],
                             'র্চ': ['চ', 'ছ', 'র'], 'র্ছ': ['র'], 'র্জ': ['জ', 'ঞ', 'র'], 'র্ঝ': ['র'], 'র্ঞ': ['জ', 'র'], 
                             'র্ট': ['ট', 'ম', 'র'], 'র্ঠ': ['র'], 'র্ড': ['র'], 'র্ঢ': ['র'], 'র্ণ': ['ড', 'ন', 'র'], 
                             'র্ত': ['ত', 'থ', 'ন', 'ব', 'ম', 'র'], 'র্থ': ['র'], 'র্দ': ['জ', 'থ', 'দ', 'ধ', 'ব', 'র'], 'র্ধ': ['ব', 'ম', 'র'], 
                             'র্ন': ['ট', 'ড', 'ত', 'দ', 'ন', 'ব', 'ম', 'স', 'র'],
                             'র্প': ['ক', 'প', 'স', 'র'], 'র্ফ': ['র'], 'র্ব': ['জ', 'ব', 'ল', 'র'], 'র্ভ': ['র'], 'র্ম': ['প', 'ব', 'ম', 'র'], 
                             'র্য': ['র'], 'র্র': ['র'], 'র্ল': ['র', 'ট', 'ড', 'স'], 'র্শ': ['চ', 'ন', 'ব', 'র'], 'র্ষ': ['ক', 'ট', 'ণ', 'প', 'ম', 'র'], 
                             'র্স': ['ক', 'চ', 'ট', 'ত', 'থ', 'প', 'ফ', 'ব', 'ম', 'র'], 'র্হ': ['র'], 'র্ড়': ['র'], 'র্ঢ়': ['র'], 'র্য়': ['র']}
                             
    
    chars = []
    i = 0
    adjust = 0
    
    while(i < len(word)):
        if i+1 < len(word) and word[i+1] == '্':
            if i+2 < len(word) and word[i+2] == 'য':
                if word[i] in consonants:
                    chars.append(word[i-adjust:i+3])
                else:
                    chars.append(word[i-adjust:i+1])
                    chars.append('্য')
                adjust = 0
                i+=3
            elif i+2 < len(word) and adjust!=0 and word[i-adjust:i+1] in forms_tripple_cluster \
                and word[i+2] in forms_tripple_cluster[word[i-adjust:i+1]]:
                if i+3 < len(word) and word[i+3] == '্':
                    adjust += 2
                    i+=2
                else:
                    chars.append(word[i-adjust:i+3])
                    adjust = 0
                    i+=3
            elif i+2 < len(word) and adjust==0 and word[i] in forms_cluster and word[i+2] in forms_cluster[word[i]]:
                if i+3 < len(word) and word[i+3] == '্':
                    adjust += 2
                    i+=2
                else:
                    chars.append(word[i-adjust:i+3])
                    adjust = 0
                    i+=3
            else:
                chars.append(word[i-adjust:i+1])
                chars.append('্')
                adjust = 0
                i+=2

        else:
            chars.append(word[i:i+1])
            i+=1

    
    #print(word)
    #print(chars)

    return chars



################################# All Diacritics Seperation #################################
def ads_grapheme_extraction(word):
    
    forms_cluster = {'ক': ['ক', 'ট', 'ত', 'ন', 'ব', 'ম', 'র', 'ল', 'ষ', 'স'],
                     'গ': ['গ', 'ধ', 'ন', 'ব', 'ম', 'ল'],
                     'ঘ': ['ন'],
                     'ঙ': ['ক', 'খ', 'গ', 'ঘ', 'ম'],
                     'চ': ['চ', 'ছ', 'ঞ'],
                     'জ': ['জ', 'ঝ', 'ঞ', 'ব'],
                     'ঞ': ['চ', 'ছ', 'জ', 'ঝ'],
                     'ট': ['ট', 'ব'],
                     'ড': ['ড'],
                     'ণ': ['ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ব', 'ম'],
                     'ত': ['ত', 'থ', 'ন', 'ব', 'ম', 'র'],
                     'থ': ['ব'],
                     'দ': ['গ', 'ঘ', 'দ', 'ধ', 'ব', 'ভ', 'ম'],
                     'ধ': ['ন', 'ব'],
                     'ন': ['জ', 'ট', 'ঠ', 'ড', 'ত', 'থ', 'দ', 'ধ', 'ন', 'ব', 'ম', 'স'],
                     'প': ['ট', 'ত', 'ন', 'প', 'ল', 'স'],
                     'ফ': ['ট', 'ল'],
                     'ব': ['জ', 'দ', 'ধ', 'ব', 'ভ', 'ল'],
                     'ভ': ['র'],
                     'ম': ['ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'ল'],
                     'ল': ['ক', 'গ', 'ট', 'ড', 'প', 'ফ', 'ব', 'ম', 'ল', 'স'],
                     'শ': ['চ', 'ছ', 'ত', 'ন', 'ব', 'ম', 'ল'],
                     'ষ': ['ক', 'ট', 'ঠ', 'ণ', 'প', 'ফ', 'ব', 'ম'],
                     'স': ['ক', 'খ', 'ট', 'ত', 'থ', 'ন', 'প', 'ফ', 'ব', 'ম', 'ল'],
                     'হ': ['ণ', 'ন', 'ব', 'ম', 'ল'],
                     'ড়': ['গ']}
    
    forms_tripple_cluster = {'ক্ষ': ['ণ', 'ম'], 'ঙ্ক': ['ষ'], 'চ্ছ': ['ব'], 'জ্জ': ['ব'],
                             'ত্ত': ['ব'], 'দ্দ': ['ব'], 'দ্ধ': ['ব'], 'দ্ভ': ['র'],
                             'ন্ত': ['ব'], 'ন্দ': ['ব'], 'ম্প': ['ল'], 'ম্ভ': ['র'],
                             'ষ্ক': ['র'], 'স্ক': ['র'], 'স্ত': ['ব', 'র'], 'স্প': ['ল']}
    
    chars = []
    i = 0
    adjust = 0
    
    while(i < len(word)):
        if i+1 < len(word) and word[i+1] == '্':
            if word[i] == 'র':
                chars.append('র্')
                adjust = 0
                i+=2
            elif i+2 < len(word) and word[i+2] == 'য':
                chars.append(word[i-adjust:i+1])
                chars.append('্য')
                adjust = 0
                i+=3
            elif i+2 < len(word) and word[i+2] == 'র':
                # Treat '্র' as a seperate grapheme
                chars.append(word[i-adjust:i+1])
                chars.append('্র')
                # Keep '্র' icluded in the cluster
                # chars.append(word[i-adjust:i+3])
                if i+3 < len(word) and word[i+3] == '্' and i+4 < len(word) and word[i+4] == 'য':    
                    chars.append('্য')
                    i+=5
                else:
                    i+=3
                adjust = 0
            elif i+2 < len(word) and adjust!=0 and word[i-adjust:i+1] in forms_tripple_cluster \
                and word[i+2] in forms_tripple_cluster[word[i-adjust:i+1]]:
                if i+3 < len(word) and word[i+3] == '্':
                    adjust += 2
                    i+=2
                else:
                    chars.append(word[i-adjust:i+3])
                    adjust = 0
                    i+=3
            elif i+2 < len(word) and adjust==0 and word[i] in forms_cluster and word[i+2] in forms_cluster[word[i]]:
                if i+3 < len(word) and word[i+3] == '্':
                    adjust += 2
                    i+=2
                else:
                    chars.append(word[i-adjust:i+3])
                    adjust = 0
                    i+=3
            else:
                chars.append(word[i-adjust:i+1])
                chars.append('্')
                adjust = 0
                i+=2

        else:
            chars.append(word[i:i+1])
            i+=1

    
    #print(word)
    #print(chars)

    return chars

