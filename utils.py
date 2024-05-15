import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# Stop words
from nltk.corpus import stopwords
stop   = list(set(stopwords.words('english')))
words  = ['rs', 'weight', 'easy', 'name', 'length', 'style', 'pattern', 'high', 'height', 'contents', 'dimensions', 'discounts','fabric', 'dimension', 'discount','ideal', 'multicolor', 'content','inch', 'size', 'dimension','brand', 'design','showpiece', 'great', 'made', 'perfect', 'inchbrand', 'best','farbric','model','material', 'type', 'package', 'prices', 'product', 'warranty', 'number', 'quality', 'details', 'price', 'color', 'pack', 'general', 'sales', 'products', 'free', 'buy', 'delivery', 'genuine', 'shipping', 'cash', 'replacement', 'day', 'flipkart.com', 'guarantee', 'online', 'features', 'specifications']
stop_words = stop #+ words

# Description des variables d'un dataframe
def get_dataframe_infos(df):
    """Examine le dataframe (ou la series) 'df' fourni en paramètre et renvoit un dataframe 'df_infos' composé des variables :
       - 'Colonne' : nom des variables de df
       - 'Type' : type de la colonne
       - 'Valeurs uniques' : nombre de valeurs unique de la colonne
       - 'Valeurs manquantes' : nombre de valeurs manquantes de la colonne
       - '% valeurs manquantes' : pourcentage de valeurs manquantes de la colonne
       - 'Doublons' : nombre de valeurs non uniques (doublons) de la colonne
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    number_of_rows = df.shape[0]
    col_names = df.columns
    col_types = df.dtypes
    unique_values = df.nunique()
    missing_values = df.isnull().sum()
    non_missing_values = number_of_rows - missing_values
    duplicate_values = non_missing_values - unique_values

    df_infos = pd.DataFrame({
        'Colonne': col_names,
        'Type': col_types,
        'Valeurs uniques': unique_values,
        'Doublons': duplicate_values,
        'Valeurs manquantes': missing_values,
        '% valeurs manquantes': round((missing_values / number_of_rows) * 100, 2).astype(str) + " %"
    }).reset_index(drop=True)

    return df_infos


# Affiche l'histogramme et le boxplot d'une feature
def plot_distribution(dataframe, feature, x_label, y_label, x_tick_angle=None):
    fig, axs = plt.subplots(2, 1, figsize=(5, 4), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)  # Ajuste l'espace entre les graphiques
    
    # Histogramme
    sns.histplot(dataframe[feature], bins=50, kde=True, ax=axs[0])
    axs[0].set_title(f'Distribution {feature}', pad=10)
    axs[0].set_xlabel('')  # Supprime le label de l'axe x pour l'histogramme
    axs[0].set_ylabel(y_label)
    axs[0].grid(linewidth=0.25)
    
    # Boxplot
    sns.boxplot(x=dataframe[feature], ax=axs[1])
    axs[1].set_xlabel(x_label)
    axs[1].set_yticks([])  # Supprime les ticks de l'axe y pour le boxplot

    if x_tick_angle is not None:
        for axis in axs:
            for label in axis.get_xticklabels():
                label.set_rotation(x_tick_angle)

    plt.show()

# def tokenize_lemmatize_text(text, nouns_and_verbs_only=True, min_length=3, tokens_to_exclude=[], rejoin=True):
#     # Conversion du texte en minuscules
#     text = text.lower()
    
#     # Création d'un tokenizer avec conservation des mot avec apostrophe et tiret
#     pattern = r"\b[a-zA-Z\-\']+\b"
#     tokenizer = RegexpTokenizer(pattern)

#     # Tokenization du texte
#     words = tokenizer.tokenize(text)
   
#     # Suppression des stop-words
#     words = [word for word in words if word not in stop]

#     # Application du POS tagging (POS = Part Of Speech)
#     tagged_words = nltk.pos_tag(words)

#     # Initialisation du lemmatizer
#     lemmatizer = WordNetLemmatizer()

#     # Lemmatisation des noms communs et verbes
#     lemmatized_words = []
#     for word, tag in tagged_words:
#         if tag.startswith('NN'):  # Noms communs (NN, NNS)
#             lemma = lemmatizer.lemmatize(word, pos='n')  # Version au singulier
#         elif tag.startswith('VB'):  # Verbes (VB, VBD, VBG, VBN, VBP, VBZ)
#             lemma = lemmatizer.lemmatize(word, pos='v')  # Version à l'infinitif
#         elif nouns_and_verbs_only:
#             continue
#         else:
#             lemma = word
#         lemmatized_words.append(lemma)

#     # Suppression des mots courts
#     if min_length > 1:
#         lemmatized_words = [word for word in lemmatized_words if len(word) >= min_length]

#     # Suppression des tokens à exclure
#     if tokens_to_exclude:
#         lemmatized_words = [word for word in lemmatized_words if word not in tokens_to_exclude]

#     # Option rejoin
#     if rejoin:
#         lemmatized_words = ' '.join(lemmatized_words)

#     return lemmatized_words

def tokenize_lemmatize_text(text, exclude_stop_words=True, nouns_and_verbs_only=True, min_length=3, tokens_to_exclude=[], rejoin=True):
    # Conversion du texte en minuscules
    text = text.lower()
    
    # Création d'un tokenizer avec conservation des mot avec apostrophe et tiret
    pattern = r"\b[a-zA-Z\-\']+\b"
    tokenizer = RegexpTokenizer(pattern)

    # Tokenization du texte
    words = tokenizer.tokenize(text)

    # Suppression des stop-words
    if exclude_stop_words:
        words = [word for word in words if word not in stop_words]

    # Application du POS tagging (POS = Part Of Speech)
    tagged_words = nltk.pos_tag(words)

    # Initialisation du lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatisation des noms communs et verbes
    lemmatized_words = []
    for word, tag in tagged_words:
        if tag.startswith('NN'):  # Noms communs (NN, NNS)
            lemma = lemmatizer.lemmatize(word, pos='n')  # Version au singulier
        elif tag.startswith('VB'):  # Verbes (VB, VBD, VBG, VBN, VBP, VBZ)
            lemma = lemmatizer.lemmatize(word, pos='v')  # Version à l'infinitif
        elif nouns_and_verbs_only:
            continue
        else:
            lemma = word
        lemmatized_words.append(lemma)

    # Suppression des mots courts
    if min_length > 1:
        lemmatized_words = [word for word in lemmatized_words if len(word) >= min_length]

    # Suppression des tokens à exclure
    if tokens_to_exclude:
        lemmatized_words = [word for word in lemmatized_words if word not in tokens_to_exclude]
    
    # Option rejoin
    if rejoin:
        lemmatized_words = ' '.join(lemmatized_words)

    return lemmatized_words