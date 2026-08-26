from pydantic import BaseModel, Field
import datasets
import pycountry


# example rows:
# fra_Latn
# [BouquetRow(uniq_id='P451-S2', src_text='Êtes-vous intéressé par le secteur technologique ?', tgt_lang='eng_Latn', domain='web misc.', par_comment='This paragraph exemplifies a job posting on a social media platform or online bulletin board. It contains informal language identifiable by interrogative and exclamatory sentences. It also includes a common strategy for gender neutral language. The second person is generic and refers to anyone interested in the job post. The first person plural is exclusive and refers to the company', orig_text='¿Tienes inquietud por el sector tecnológico?', has_speaker_tag=False, tgt_text='Are you interested in the technology sector?', has_hashtag=False, has_emoji=False, has_12p=True, newline_next=True, tags='<na>', register='usa', par_id='P451', sent_comment=None, split='dev', level='sentence_level', src_lang='fra_Latn'),
#  BouquetRow(uniq_id='P090-S5', src_text='Par contre, en tant que film avec un thème local de Hong Kong, il a atteint un certain équilibre entre les caractéristiques de niche et le marché.', tgt_lang='eng_Latn', domain='social posts', par_comment='Blog of a movie review', orig_text='不过，作为香港本土题材的电影，它在小众特色与市场之间取得了一定平衡 。', has_speaker_tag=False, tgt_text='However, as a film with a local Hong Kong theme, it has achieved a certain balance between niche characteristics and the market.', has_hashtag=False, has_emoji=False, has_12p=False, newline_next=False, tags='phrase:normalisation', register='nca', par_id='P090', sent_comment=None, split='dev', level='sentence_level', src_lang='fra_Latn'),
#  BouquetRow(uniq_id='P194-S2', src_text="La salle de réception dans laquelle Diane pénétra baignait dans une lumière froide et avait été affublée de couleurs chatoyantes sans le moindre égard pour l'harmonie.", tgt_lang='eng_Latn', domain='narration', par_comment='Imaginary description of a scene in a novel (probably set in the past) in which a young woman is compelled by her father to go to a party where she will meet a young man who wants to approach her. Diane is of the feminine grammatical gender; Aymeric of the masculine grammatical gender.', orig_text="La salle de réception dans laquelle Diane pénétra baignait dans une lumière froide et avait été affublée de couleurs chatoyantes sans le moindre égard pour l'harmonie.", has_speaker_tag=False, tgt_text='The reception hall where Diane entered was bathed in cold light and had been decked with sparkling colors without any regard for harmony.', has_hashtag=False, has_emoji=False, has_12p=False, newline_next=False, tags='clause:comparative, clause:preposition_phrase, phrase:verb_tense, phrase:adjective_order', register='ica', par_id='P194', sent_comment=None, split='dev', level='sentence_level', src_lang='fra_Latn')]
# ind_Latn
# [BouquetRow(uniq_id='P001-S1', src_text='Cara pembuatan mahshi atau dolma bervariasi dari satu negara ke negara lain.', tgt_lang='eng_Latn', domain='how-to | instructions', par_comment="The named entity 'mahshi' refers to a popular cooking term in Egyptian Arabic used for any stuffed vegetable or leaf—such as grape leaves, cabbage, or peppers. The literal meaning in Standard Arabic is 'stuffed,' and in this context, it refers to food.", orig_text='طريقة عمل المحشي بتختلف كتير من بلد لبلد', has_speaker_tag=False, tgt_text='Mahshi "dolma" recipe varies a lot from one country to another.', has_hashtag=False, has_emoji=False, has_12p=False, newline_next=False, tags='Lexical, impersonal', register='uca', par_id='P001', sent_comment=None, split='dev', level='sentence_level', src_lang='ind_Latn'),
#  BouquetRow(uniq_id='P001-S2', src_text='Di Mesir, biasanya isiannya berupa nasi dan sayuran.', tgt_lang='eng_Latn', domain='how-to | instructions', par_comment='<na>', orig_text='في مصر عادةً المحشي بيتحشي برز وخضار', has_speaker_tag=False, tgt_text="In Egypt usually it's stuffed with rice and veggies.", has_hashtag=False, has_emoji=False, has_12p=False, newline_next=False, tags='impersonal', register='uca', par_id='P001', sent_comment=None, split='dev', level='sentence_level', src_lang='ind_Latn'),
#  BouquetRow(uniq_id='P001-S3', src_text='Di kawasan Mediterania Timur, mereka juga menambahkan daging sebagai isiannya.', tgt_lang='eng_Latn', domain='how-to | instructions', par_comment='<na>', orig_text='في بلاد الشام بيحطو فيه لحمة كمان', has_speaker_tag=False, tgt_text='In the levant they stuff it with meat as well.', has_hashtag=False, has_emoji=False, has_12p=False, newline_next=False, tags='impersonal', register='uca', par_id='P001', sent_comment=None, split='dev', level='sentence_level', src_lang='ind_Latn')]
class BouquetRow(BaseModel):
    uniq_id: str
    src_text: str
    tgt_lang: str
    domain: str
    par_comment: str
    orig_text: str
    tgt_text: str
    newline_next: bool
    tags: str
    register: str
    par_id: str
    split: str
    level: str
    src_lang: str
    

    @classmethod
    def load(cls, lang, split) -> list["BouquetRow"]:
        """ Load the BOUQuET dataset for a given language and split. 
        Args:
            lang: Language code, e.g., "fra_Latn"
            split: Dataset split, e.g., "dev", "test"
        Returns:
            List of BouquetRow instances.
        """
        dataset = datasets.load_dataset("facebook/bouquet", lang, split=split)
        return [cls.model_validate(row) for row in dataset]


def parse_bool(value: object, default: bool = False) -> bool:
    """Parse a best-effort boolean from model output."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if value is None:
        return default
    return bool(value)


def get_lang_name(lang_code: str) -> str:
    """Extract language name from FLORES-200 code (e.g., fra_Latn -> French)."""
    # Extract ISO code from FLORES format (e.g., "fra" from "fra_Latn")
    iso_code = lang_code.split("_")[0]
    
    # Try to find language by 2-letter or 3-letter code
    if len(iso_code) == 2:
        lang = pycountry.languages.get(alpha_2=iso_code)
    else:
        lang = pycountry.languages.get(alpha_3=iso_code)
    
    if not lang:
        raise ValueError(f"Could not find language name for code: {lang_code}")
    return lang.name

def get_madlad_lang(bouquet_lang: str) -> str:
    """Get MADLAD language code from BOUQuET language code."""
    iso_code = bouquet_lang.split("_")[0]
    lang = pycountry.languages.get(alpha_2=iso_code)
    if not lang:
        lang = pycountry.languages.get(alpha_3=iso_code)
    if not lang:
        raise ValueError(f"Could not find language code for {bouquet_lang}")
    return lang.alpha_2
