"""
Translate WMT24++ with vLLM, with or without user instructions.

Usage:
    python -m translate.wmt24pp_vllm --model google/gemma-3-4b-it --lang en-fr_FR
    python -m translate.wmt24pp_vllm --model google/gemma-3-4b-it --lang en-fr_FR --use-user-instructions
    python -m translate.wmt24pp_vllm --model google/gemma-3-27b-it --lang en-fr_FR --use-user-instructions
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import re
from typing import List

from sacrebleu import corpus_chrf

from common.wmt24pp_utils import (
    Wmt24ppDocumentGroup,
    Wmt24ppFewShotExample,
    Wmt24ppInstructionRow,
)
from translate.bouquet_vllm import (
    FewShotExample,
    _build_numbered_list_messages,
    resolve_vllm_model_and_lora,
)


# WMT24++ target locale prefix -> BOUQuET language code, for sourcing
# sentence-in-context few-shot examples in --paragraph-level mode.
WMT_TO_BOUQUET_LANG = {"fr": "fra_Latn", "id": "ind_Latn", "uk": "ukr_Cyrl"}


# 3 fixed few-shot examples per target language (hand-picked from WMT24++ domains)
_SOCIAL_EN = "y'all watching too much tv. enjoy your quarantine lol"
_SOCIAL_INSTR = "Translate as an informal social media post. Use casual, colloquial language with internet-native phrasing."
_NEWS_EN = "Siso's depictions of land, water center new gallery exhibition"
_NEWS_INSTR = "Translate as a news headline for a general readership. Use a neutral, informative journalistic tone."
_LITERARY_EN = "The fruit trees in the small garden behind the house had long since stopped bearing, but the roses, roses and more roses still gave off a velvety warmth."
_LITERARY_INSTR = "Translate as literary fiction with rich, evocative prose. Maintain the poetic imagery of the original."

FEWSHOT_EXAMPLES_BY_LANG = {
    "fr": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="vous regardez trop la télé. profitez de votre quarantaine lol"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Exposition des travaux de Siso sur la terre et l'eau, à la nouvelle galerie du centre"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Les arbres fruitiers du petit jardin derrière la maison avaient cessé de produire depuis longtemps, mais les roses, les roses et encore les roses dégageaient toujours une chaleur veloutée."),
    ],
    "id": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="kalian kebanyakan nonton tv. nikmatin aja karantina kalian wkwk"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Penggambaran tanah dan air karya Siso menjadi pusat pameran galeri baru"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Pohon-pohon buah di taman kecil di belakang rumah sudah lama berhenti berbuah, tetapi mawar, mawar dan lebih banyak mawar masih memancarkan kehangatan yang lembut bagai beludru."),
    ],
    "uk": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="ви занадто багато дивитесь телік. насолоджуйтесь своїм карантином лол"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Зображення землі та води Сісо стали центром нової виставки в галереї"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Фруктові дерева в маленькому саду за будинком давно перестали плодоносити, але троянди, троянди і ще раз троянди все ще випромінювали оксамитове тепло."),
    ],
    "zu": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="Nibukela i-TV kakhulu. Jabulelani ukuhlala kwenu bodwa lol"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Izithombe zikaSiso zomhlaba, umbukiso wegalari entsha yesikhungo samanzi"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Izihlahla zezithelo engadini encane ngemuva kwendlu zazikade zingasatheli, kodwa ama-rose, ama-rose kanye nama-rose amaningi asanikeza ukufudumala okuthambile."),
    ],
    "gu": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="તમે બધા ખૂબ ટીવી જુઓ છો. તમારા ક્વોરન્ટાઇનનો આનંદ માણો lol"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="જમીન, પાણી કેન્દ્રના નવા ગેલેરી પ્રદર્શનનું સિસોનું ચિત્રણ"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="ઘરની પાછળના નાના બગીચામાં ફળ આપતા ઝાડ ઘણા સમયથી બંધ થઈ ગયા હતા, પરંતુ ગુલાબ, ગુલાબ અને વધુ ગુલાબ હજુ પણ મખમલી હૂંફ આપતા હતા."),
    ],
    "th": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="พวกคุณดูทีวีมากเกินไปแล้ว สนุกกับการกักตัวกันเถอะ ฮ่าๆ"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="ภาพวาดของซิโซที่แสดงถึงผืนดินและผืนน้ำเป็นหัวใจสำคัญของนิทรรศการในแกลเลอรี่ใหม่"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="ต้นไม้ผลในสวนเล็กๆ หลังบ้านหยุดออกผลไปนานแล้ว แต่ดอกกุหลาบ ดอกกุหลาบ และดอกกุหลาบอีกมากมายยังคงส่งความอบอุ่นนุ่มนวลออกมา"),
    ],
    "zh": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="你们电视看太多了吧 好好享受隔离吧哈哈"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="西索描绘土地与水的作品成为新画廊展览的焦点"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="房子后面小花园里的果树早已不再结果，但那些玫瑰，玫瑰，还有更多的玫瑰，依然散发着天鹅绒般的温暖。"),
    ],
    "ko": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="너네 TV 너무 많이 본다. 자가격리 즐겨 ㅋㅋ"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="시소의 대지와 물 묘사, 새 갤러리 전시의 중심에 서다"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="집 뒤 작은 정원의 과일나무들은 오래전에 열매를 맺지 않았지만, 장미, 장미, 그리고 더 많은 장미들은 여전히 벨벳 같은 온기를 내뿜고 있었다."),
    ],
    "hi": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="तुम लोग बहुत ज़्यादा टीवी देखते हो। अपने क्वारंटीन का मज़ा लो lol"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="सिसो द्वारा ज़मीन और पानी का चित्रण नई गैलरी प्रदर्शनी का केंद्र बना"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="घर के पीछे छोटे बगीचे में फलों के पेड़ बहुत पहले फल देना बंद कर चुके थे, लेकिन गुलाब, गुलाब और और भी गुलाब अभी भी मखमली गर्माहट बिखेर रहे थे।"),
    ],
    "ar": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="إنتو بتتفرجوا على التلفزيون كتير. استمتعوا بالحجر الصحي بتاعكم هههه"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="تصوير سيسو للأرض والماء في صدارة معرض الغاليري الجديد"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="توقفت أشجار الفاكهة في الحديقة الصغيرة خلف المنزل عن الإثمار منذ زمن بعيد، لكن الورود، الورود ثم المزيد من الورود كانت لا تزال تنبعث منها دفء مخملي."),
    ],
    "vi": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="mấy bạn xem TV nhiều quá đi. tận hưởng kỳ cách ly đi nha haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Tranh vẽ đất và nước của Siso trở thành tâm điểm triển lãm mới tại phòng trưng bày"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Những cây ăn quả trong khu vườn nhỏ phía sau nhà đã ngừng ra trái từ lâu, nhưng hoa hồng, hoa hồng và thêm nữa hoa hồng vẫn tỏa ra hơi ấm mượt mà như nhung."),
    ],
    "ja": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="みんなテレビ見すぎでしょ。隔離生活楽しんで笑"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="シソの大地と水の描写、新ギャラリー展の中心に"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="家の裏の小さな庭の果樹はとうに実をつけなくなっていたが、バラ、バラ、そしてさらに多くのバラが、いまだにビロードのような温もりを放っていた。"),
    ],
    "de": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="ihr guckt echt zu viel Fernsehen. genießt eure Quarantäne lol"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sisos Darstellungen von Land und Wasser im Mittelpunkt neuer Galerieausstellung"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Die Obstbäume im kleinen Garten hinter dem Haus trugen schon lange keine Früchte mehr, doch die Rosen, Rosen und noch mehr Rosen verströmten noch immer eine samtene Wärme."),
    ],
    "pt": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="vocês assistem TV demais. aproveitem a quarentena de vocês kk"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Representações de terra e água de Siso são destaque em nova exposição da galeria"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="As árvores frutíferas no pequeno jardim atrás da casa já haviam parado de dar frutos há muito tempo, mas as rosas, rosas e mais rosas ainda exalavam um calor aveludado."),
    ],
    "tr": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="çok fazla TV izliyorsunuz. karantinanızın tadını çıkarın sjsj"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Siso'nun toprak ve su tasvirleri yeni galeri sergisinin odağında"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Evin arkasındaki küçük bahçedeki meyve ağaçları çoktan meyve vermez olmuştu, ama güller, güller ve daha fazla güller hâlâ kadifemsi bir sıcaklık yayıyordu."),
    ],
    "sw": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="mnaangalia TV sana. furahieni karantini yenu haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Picha za Siso za ardhi na maji katikati ya maonyesho mapya ya nyumba ya sanaa"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Miti ya matunda katika bustani ndogo nyuma ya nyumba ilikuwa imesimama kuzaa zamani, lakini waridi, waridi na waridi zaidi bado zilitoa joto laini kama hariri."),
    ],
    "ru": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="вы слишком много телик смотрите. наслаждайтесь карантином лол"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Изображения земли и воды Сисо в центре новой выставки галереи"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Фруктовые деревья в маленьком саду за домом давно перестали плодоносить, но розы, розы и ещё раз розы по-прежнему источали бархатное тепло."),
    ],
    "es": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="ven demasiada tele. disfruten su cuarentena jaja"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Las representaciones de tierra y agua de Siso son el centro de la nueva exposición de la galería"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Los árboles frutales del pequeño jardín detrás de la casa hacía mucho que habían dejado de dar frutos, pero las rosas, rosas y más rosas aún desprendían una calidez aterciopelada."),
    ],
    "it": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="guardate troppa TV. godetevi la quarantena ahah"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Le raffigurazioni di terra e acqua di Siso al centro della nuova mostra della galleria"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Gli alberi da frutto nel piccolo giardino dietro la casa avevano smesso di produrre da tempo, ma le rose, rose e ancora rose emanavano ancora un calore vellutato."),
    ],
    "pl": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="za dużo telewizji oglądacie. cieszcie się kwarantanną haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Przedstawienia ziemi i wody Siso w centrum nowej wystawy galerii"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Drzewa owocowe w małym ogrodzie za domem dawno przestały owocować, ale róże, róże i jeszcze więcej róż wciąż roztaczały aksamitne ciepło."),
    ],
    "bn": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="তোমরা বেশি টিভি দেখো। কোয়ারেন্টাইন উপভোগ করো হাহা"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="সিসোর ভূমি ও জলের চিত্রায়ণ নতুন গ্যালারি প্রদর্শনীর কেন্দ্রবিন্দু"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="বাড়ির পিছনের ছোট্ট বাগানের ফলের গাছগুলো অনেক আগেই ফল দেওয়া বন্ধ করে দিয়েছিল, কিন্তু গোলাপ, গোলাপ আর আরও গোলাপ এখনও মখমলের মতো উষ্ণতা ছড়িয়ে দিচ্ছিল।"),
    ],
    "fa": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="شما خیلی زیاد تلویزیون می‌بینید. از قرنطینه‌تون لذت ببرید هها"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="تصویرسازی‌های سیسو از زمین و آب در مرکز نمایشگاه جدید گالری"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="درختان میوه در باغ کوچک پشت خانه مدت‌ها بود که دیگر میوه نمی‌دادند، اما گل‌های رز، رز و باز هم رز همچنان گرمایی مخملین ساطع می‌کردند."),
    ],
    "fil": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="ang dami nyong pinapanood na TV. enjoy nyo quarantine nyo haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Mga paglalarawan ni Siso ng lupa at tubig, sentro ng bagong eksibisyon sa gallery"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Matagal nang hindi namumunga ang mga punong-prutas sa maliit na hardin sa likod ng bahay, ngunit ang mga rosas, rosas at higit pang mga rosas ay nagbibigay pa rin ng mainit na pakiramdam na parang belbet."),
    ],
    "he": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="אתם רואים יותר מדי טלוויזיה. תהנו מהבידוד שלכם חחח"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="תיאורי הקרקע והמים של סיסו במרכז תערוכת הגלריה החדשה"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="עצי הפרי בגינה הקטנה שמאחורי הבית כבר מזמן חדלו להניב פירות, אבל הוורדים, ורדים ועוד ורדים עדיין הפיצו חום קטיפתי."),
    ],
    "ta": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="நீங்க ரொம்ப TV பாக்குறீங்க. உங்க quarantine-ஐ enjoy பண்ணுங்க ஹஹ"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="நிலம், நீர் குறித்த சிசோவின் சித்தரிப்புகள் புதிய கேலரி கண்காட்சியின் மையமாக"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="வீட்டின் பின்புறம் உள்ள சிறிய தோட்டத்தில் பழ மரங்கள் நீண்ட காலமாக காய்க்காமல் நின்றுவிட்டன, ஆனால் ரோஜாக்கள், ரோஜாக்கள், இன்னும் ரோஜாக்கள் இன்னும் பட்டுப்போன்ற இளஞ்சூடு பரப்பின."),
    ],
    "el": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="βλέπετε πολλή τηλεόραση. απολαύστε την καραντίνα σας χαχα"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Οι απεικονίσεις γης και νερού του Σίσο στο επίκεντρο νέας έκθεσης γκαλερί"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Τα οπωροφόρα δέντρα στον μικρό κήπο πίσω από το σπίτι είχαν πάψει να καρποφορούν εδώ και καιρό, αλλά τα τριαντάφυλλα, τριαντάφυλλα και ακόμη περισσότερα τριαντάφυλλα εξακολουθούσαν να αναδίδουν μια βελούδινη ζεστασιά."),
    ],
    "cs": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="koukáte moc na televizi. užijte si karanténu haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sisovy obrazy země a vody v centru nové výstavy galerie"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Ovocné stromy v malé zahradě za domem už dávno přestaly plodit, ale růže, růže a ještě více růží stále vydávaly sametové teplo."),
    ],
    "nl": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="jullie kijken veel te veel tv. geniet van je quarantaine haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Siso's afbeeldingen van land en water centraal in nieuwe galerie-expositie"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="De fruitbomen in de kleine tuin achter het huis waren al lang gestopt met dragen, maar de rozen, rozen en nog meer rozen verspreidden nog steeds een fluwelen warmte."),
    ],
    "ro": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="vă uitați prea mult la televizor. bucurați-vă de carantină haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Reprezentările lui Siso ale pământului și apei, în centrul noii expoziții a galeriei"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Pomii fructiferi din grădina mică din spatele casei încetaseră de mult să mai rodească, dar trandafirii, trandafiri și încă mai mulți trandafiri răspândeau încă o căldură catifelată."),
    ],
    "hu": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="túl sokat tévéztek. élvezzétek a karantént haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Siso föld- és vízábrázolásai az új galériakiállítás középpontjában"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="A ház mögötti kis kert gyümölcsfái már régen abbahagyták a termést, de a rózsák, rózsák és még több rózsa még mindig bársonyos melegséget árasztott."),
    ],
    "sv": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="ni tittar för mycket på tv. njut av er karantän haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sisos skildringar av land och vatten i centrum för ny galleriutställning"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Fruktträden i den lilla trädgården bakom huset hade för länge sedan slutat bära frukt, men rosorna, rosor och ännu fler rosor spred fortfarande en sammetslen värme."),
    ],
    "te": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="మీరు చాలా ఎక్కువ TV చూస్తున్నారు. మీ క్వారంటైన్ ఎంజాయ్ చేయండి హహ"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="భూమి, నీటిపై సిసో చిత్రణలు కొత్త గ్యాలరీ ప్రదర్శన కేంద్రంగా"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="ఇంటి వెనుక చిన్న తోటలోని పండ్ల చెట్లు చాలా కాలం క్రితమే కాయలు కాయడం మానేశాయి, కానీ గులాబీలు, గులాబీలు మరియు మరిన్ని గులాబీలు ఇప్పటికీ వెల్వెట్ లాంటి వెచ్చదనాన్ని వెదజల్లుతున్నాయి."),
    ],
    "ur": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="تم لوگ بہت زیادہ ٹی وی دیکھتے ہو۔ اپنی قرنطینہ سے لطف اٹھاؤ ہاہا"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="سیسو کی زمین اور پانی کی تصویر کشی نئی گیلری نمائش کا مرکز"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="گھر کے پیچھے چھوٹے باغ میں پھلوں کے درختوں نے بہت پہلے پھل دینا بند کر دیا تھا، لیکن گلاب، گلاب اور مزید گلاب اب بھی مخملی گرمی بکھیر رہے تھے۔"),
    ],
    "sr": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="превише гледате ТВ. уживајте у карантину хаха"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Сисове слике земље и воде у центру нове изложбе галерије"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Воћке у малом врту иза куће одавно су престале да рађају, али руже, руже и још руже и даље су одавале баршунасту топлину."),
    ],
    "ca": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="mireu massa la tele. gaudiu de la quarantena jaja"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Les representacions de terra i aigua de Siso, al centre de la nova exposició de la galeria"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Els arbres fruiters del petit jardí darrere la casa feia temps que havien deixat de donar fruit, però les roses, roses i més roses encara desprenien una escalfor vellutada."),
    ],
    "no": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="dere ser altfor mye på TV. nyt karantenen deres haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sisos skildringer av land og vann i sentrum for ny galleriutstilling"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Frukttrærne i den lille hagen bak huset hadde for lengst sluttet å bære frukt, men rosene, roser og enda flere roser spredte fortsatt en fløyelsmyk varme."),
    ],
    "bg": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="гледате прекалено много телевизия. наслаждавайте се на карантината хаха"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Изображенията на земя и вода на Сисо в центъра на нова изложба в галерията"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Плодните дървета в малката градина зад къщата отдавна бяха спрели да раждат, но розите, рози и още повече рози все още излъчваха кадифена топлина."),
    ],
    "da": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="I ser alt for meget TV. nyd jeres karantæne haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sisos skildringer af land og vand i centrum for ny galleriudstilling"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Frugttræerne i den lille have bag huset havde for længst holdt op med at bære frugt, men roserne, roser og endnu flere roser udstrålede stadig en fløjlsblød varme."),
    ],
    "fi": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="katotte liikaa telkkaa. nauttikaa karanteenista haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sison maan ja veden kuvaukset uuden gallerianäyttelyn keskipisteenä"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Talon takana olevan pienen puutarhan hedelmäpuut olivat jo kauan sitten lakanneet kantamasta hedelmää, mutta ruusut, ruusut ja yhä useammat ruusut levittivät yhä samettista lämpöä."),
    ],
    "kn": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="ನೀವು ತುಂಬಾ TV ನೋಡ್ತೀರಿ. ನಿಮ್ಮ ಕ್ವಾರಂಟೈನ್ ಎಂಜಾಯ್ ಮಾಡಿ ಹಹ"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="ಭೂಮಿ ಮತ್ತು ನೀರಿನ ಸಿಸೊ ಅವರ ಚಿತ್ರಣಗಳು ಹೊಸ ಗ್ಯಾಲರಿ ಪ್ರದರ್ಶನದ ಕೇಂದ್ರಬಿಂದು"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="ಮನೆಯ ಹಿಂದಿನ ಸಣ್ಣ ತೋಟದಲ್ಲಿನ ಹಣ್ಣಿನ ಮರಗಳು ಬಹಳ ಹಿಂದೆಯೇ ಫಲ ಕೊಡುವುದನ್ನು ನಿಲ್ಲಿಸಿದ್ದವು, ಆದರೆ ಗುಲಾಬಿಗಳು, ಗುಲಾಬಿಗಳು ಮತ್ತು ಇನ್ನಷ್ಟು ಗುಲಾಬಿಗಳು ಇನ್ನೂ ಮೆತ್ತನೆಯ ಬೆಚ್ಚನೆಯ ಕಾಂತಿಯನ್ನು ಹರಡುತ್ತಿದ್ದವು."),
    ],
    "ml": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="നിങ്ങൾ ടിവി കൂടുതൽ കാണുന്നുണ്ട്. ക്വാറന്റൈൻ ആസ്വദിക്കൂ ഹഹ"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="ഭൂമിയുടെയും ജലത്തിന്റെയും സിസോയുടെ ചിത്രീകരണങ്ങൾ പുതിയ ഗാലറി പ്രദർശനത്തിന്റെ കേന്ദ്രബിന്ദു"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="വീടിന്റെ പിന്നിലെ ചെറിയ തോട്ടത്തിലെ ഫലവൃക്ഷങ്ങൾ വളരെ മുമ്പേ കായ്ക്കുന്നത് നിർത്തിയിരുന്നു, എന്നാൽ റോസാപ്പൂക്കൾ, റോസാപ്പൂക്കൾ, കൂടുതൽ റോസാപ്പൂക്കൾ ഇപ്പോഴും മൃദുലമായ ചൂട് പരത്തിക്കൊണ്ടിരുന്നു."),
    ],
    "mr": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="तुम्ही खूप टीव्ही बघता. तुमच्या क्वारंटाईनचा आनंद घ्या हाहा"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="सिसोचे जमीन आणि पाण्याचे चित्रण नवीन गॅलरी प्रदर्शनाच्या केंद्रस्थानी"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="घराच्या मागच्या छोट्या बागेतील फळझाडांनी खूप पूर्वीच फळ देणे बंद केले होते, पण गुलाब, गुलाब आणि आणखी गुलाब अजूनही मखमली उबदारपणा पसरवत होते."),
    ],
    "pa": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="ਤੁਸੀਂ ਬਹੁਤ ਜ਼ਿਆਦਾ ਟੀਵੀ ਦੇਖਦੇ ਹੋ। ਆਪਣੀ ਕੁਆਰੰਟੀਨ ਦਾ ਮਜ਼ਾ ਲਓ ਹਾਹਾ"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="ਸਿਸੋ ਦੀਆਂ ਜ਼ਮੀਨ ਅਤੇ ਪਾਣੀ ਦੀਆਂ ਤਸਵੀਰਾਂ ਨਵੀਂ ਗੈਲਰੀ ਪ੍ਰਦਰਸ਼ਨੀ ਦਾ ਕੇਂਦਰ"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="ਘਰ ਦੇ ਪਿੱਛੇ ਛੋਟੇ ਬਾਗ਼ ਵਿੱਚ ਫਲਾਂ ਦੇ ਰੁੱਖ ਬਹੁਤ ਪਹਿਲਾਂ ਫਲ ਦੇਣਾ ਬੰਦ ਕਰ ਚੁੱਕੇ ਸਨ, ਪਰ ਗੁਲਾਬ, ਗੁਲਾਬ ਅਤੇ ਹੋਰ ਗੁਲਾਬ ਅਜੇ ਵੀ ਮਖ਼ਮਲੀ ਨਿੱਘ ਬਿਖੇਰ ਰਹੇ ਸਨ।"),
    ],
    "pt_PT": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="vocês veem televisão a mais. aproveitem a vossa quarentena haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Representações de terra e água de Siso no centro de nova exposição da galeria"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="As árvores de fruto no pequeno jardim atrás da casa já há muito tinham parado de dar frutos, mas as rosas, rosas e mais rosas ainda exalavam um calor aveludado."),
    ],
    "sk": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="pozeráte príliš veľa televízie. užívajte si karanténu haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sisove zobrazenia zeme a vody v centre novej výstavy galérie"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Ovocné stromy v malej záhrade za domom už dávno prestali rodiť, ale ruže, ruže a ešte viac ruží stále vydávali zamatové teplo."),
    ],
    "sl": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="preveč gledate televizijo. uživajte v karanteni haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sisove upodobitve zemlje in vode v središču nove galerijske razstave"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Sadna drevesa v majhnem vrtu za hišo so že zdavnaj prenehala roditi, a vrtnice, vrtnice in še več vrtnic so še vedno oddajale žametno toploto."),
    ],
    "ar_SA": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="أنتم تشوفون تلفزيون وايد. استمتعوا بالحجر الصحي حقكم ههه"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="تصوير سيسو للأرض والماء في صدارة معرض الغاليري الجديد"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="توقفت أشجار الفاكهة في الحديقة الصغيرة خلف المنزل عن الإثمار منذ زمن بعيد، لكن الورود، الورود ثم المزيد من الورود كانت لا تزال تنبعث منها دفء مخملي."),
    ],
    "et": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="te vaatate liiga palju telekat. nautige oma karantiini haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Siso maa ja vee kujutised uue galeriinäituse keskmes"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Maja taga asuva väikese aia viljapuud olid ammu lõpetanud vilja kandmise, kuid roosid, roosid ja veel roosid kiirgasid endiselt sametist soojust."),
    ],
    "fr_CA": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="vous regardez ben trop la TV. profitez de votre quarantaine lol"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Les représentations de la terre et de l'eau de Siso au centre d'une nouvelle exposition de la galerie"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Les arbres fruitiers dans le petit jardin derrière la maison avaient cessé de produire depuis longtemps, mais les roses, les roses et encore des roses dégageaient toujours une chaleur veloutée."),
    ],
    "hr": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="previše gledate TV. uživajte u karanteni haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Sisovi prikazi zemlje i vode u središtu nove izložbe galerije"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Voćke u malom vrtu iza kuće odavno su prestale rađati, ali ruže, ruže i još više ruža i dalje su odavale baršunastu toplinu."),
    ],
    "is": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="þið horfið of mikið á sjónvarpið. njótið sóttkvíarinnar haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Lýsingar Siso á landi og vatni í brennidepli nýrrar gallerísýningar"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Ávaxtatréin í litla garðinum að baki húsinu höfðu löngu hætt að bera ávöxt, en rósirnar, rósir og enn fleiri rósir gáfu enn frá sér flauelsmjúkan hlýja."),
    ],
    "lt": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="per daug žiūrite televizorių. mėgaukitės karantinu haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Siso žemės ir vandens vaizdai naujosios galerijos parodos centre"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Vaismedžiai mažame sode už namo jau seniai nustojo derėti, bet rožės, rožės ir dar daugiau rožių vis dar skleidė aksominę šilumą."),
    ],
    "lv": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="jūs skatāties pārāk daudz TV. izbaudiet savu karantīnu haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Siso zemes un ūdens attēlojumi jaunās galerijas izstādes centrā"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Augļu koki mazajā dārzā aiz mājas jau sen bija pārstājuši nest augļus, bet rozes, rozes un vēl vairāk rožu joprojām izstaroja samtainu siltumu."),
    ],
    "sw_TZ": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="mnaangalia TV sana. furahieni karantini yenu haha"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="Picha za Siso za ardhi na maji katikati ya maonyesho mapya ya nyumba ya sanaa"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="Miti ya matunda katika bustani ndogo nyuma ya nyumba ilikuwa imesimama kuzaa zamani, lakini waridi, waridi na waridi zaidi bado zilitoa joto laini kama hariri."),
    ],
    "zh_TW": [
        Wmt24ppFewShotExample(en=_SOCIAL_EN, instruction=_SOCIAL_INSTR,
            tgt="你們電視看太多了吧 好好享受隔離吧哈哈"),
        Wmt24ppFewShotExample(en=_NEWS_EN, instruction=_NEWS_INSTR,
            tgt="西索描繪土地與水的作品成為新畫廊展覽的焦點"),
        Wmt24ppFewShotExample(en=_LITERARY_EN, instruction=_LITERARY_INSTR,
            tgt="房子後面小花園裡的果樹早已不再結果，但那些玫瑰，玫瑰，還有更多的玫瑰，依然散發著天鵝絨般的溫暖。"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Translate WMT24++ with vLLM")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        help="LLM to use for translation")
    parser.add_argument("--lang", type=str, default="en-fr_FR",
                        help="WMT24++ language pair (default: en-fr_FR)")
    parser.add_argument("--use-user-instructions", action="store_true",
                        help="Include user instructions in the prompt")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of rows to translate (None = all)")
    parser.add_argument("--ignore-cache", action="store_true",
                        help="Ignore existing results and re-translate")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking/reasoning mode (Gemma 4). Saves reasoning traces alongside translations.")
    parser.add_argument("--paragraph-level", action="store_true",
                        help="Sentence-in-context translation: feed the full document with the target sentence wrapped in [START_TRANSLATE]...[END_TRANSLATE] markers.")
    parser.add_argument("--numbered-lists", action="store_true",
                        help="P2 strategy: translate the whole document at once as an S1:/S2:/... numbered list (requires --paragraph-level).")
    parser.add_argument("--instruction-type", type=str, default="full",
                        help="Instruction variant: 'full' (curated wmt24pp_instructions.jsonl) "
                             "or e.g. 'self-para' (wmt24pp_instructions_self-para.jsonl). "
                             "A non-full value implies --use-user-instructions.")
    return parser.parse_args()


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    return corpus_chrf(predictions, [references], word_order=2).score


def evaluate_translations(rows: List[Wmt24ppInstructionRow]) -> dict:
    """Compute ChrF++ scores overall and per domain."""
    valid_rows = [r for r in rows if r.pred_text is not None]
    if not valid_rows:
        return {"overall": 0.0, "by_domain": {}}

    predictions = [r.pred_text or "" for r in valid_rows]
    references = [r.target for r in valid_rows]
    overall = compute_chrf(predictions, references)

    domains = set(r.domain for r in valid_rows)
    by_domain = {}
    for domain in sorted(domains):
        domain_rows = [r for r in valid_rows if r.domain == domain]
        domain_preds = [r.pred_text or "" for r in domain_rows]
        domain_refs = [r.target for r in domain_rows]
        by_domain[domain] = {
            "chrf": compute_chrf(domain_preds, domain_refs),
            "count": len(domain_rows),
        }

    return {"overall": overall, "by_domain": by_domain}


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    args = parse_args()

    # --instruction-type implies --use-user-instructions
    if args.instruction_type != "full":
        args.use_user_instructions = True

    print(f"Model: {args.model}")
    print(f"Language: {args.lang}")
    print(f"Use instructions: {args.use_user_instructions}")
    print(f"Instruction type: {args.instruction_type}")
    print(f"Enable thinking: {args.enable_thinking}")
    print(f"Paragraph level: {args.paragraph_level}")
    print(f"Numbered lists: {args.numbered_lists}")
    print(f"Limit: {args.limit}")

    if args.numbered_lists and not args.paragraph_level:
        raise ValueError("--numbered-lists requires --paragraph-level")

    # Check cache
    savepath = Wmt24ppInstructionRow.get_savepath(
        args.model, args.lang, args.use_user_instructions,
        enable_thinking=args.enable_thinking, paragraph_level=args.paragraph_level,
        numbered_lists=args.numbered_lists, instruction_type=args.instruction_type,
    )
    if savepath.exists() and not args.ignore_cache:
        print(f"\nResults already exist at {savepath}")
        print("Use --ignore-cache to re-translate")
        rows = Wmt24ppInstructionRow.load(savepath)
        scores = evaluate_translations(rows)
        print(f"\nOverall ChrF++: {scores['overall']:.2f}")
        for domain, stats in scores["by_domain"].items():
            print(f"  {domain}: {stats['chrf']:.2f} (n={stats['count']})")
        return

    # Load data
    rows = Wmt24ppInstructionRow.load_for_lang(
        args.lang, args.limit, instruction_type=args.instruction_type,
    )
    print(f"\nLoaded {len(rows)} rows")

    # Build messages
    import pycountry
    lang_locale = args.lang.split("-", 1)[1]  # "en-fr_FR" -> "fr_FR"
    lang_prefix = lang_locale.split("_")[0]  # "fr_FR" -> "fr"
    _py_lang = pycountry.languages.get(alpha_2=lang_prefix)
    lang_name = _py_lang.name if _py_lang else lang_locale  # for numbered-list system prompt
    doc_groups = None  # set in paragraph modes; reused at output-parsing time
    if args.paragraph_level and args.numbered_lists:
        # P2: translate the whole document at once as an S1:/S2:/... numbered list.
        bouquet_lang = WMT_TO_BOUQUET_LANG.get(lang_prefix)
        if bouquet_lang is None:
            raise ValueError(
                f"--numbered-lists needs BOUQuET numbered-list few-shot examples, "
                f"available only for {sorted(WMT_TO_BOUQUET_LANG)}, got '{lang_prefix}'"
            )
        few_shot_examples = FewShotExample.build_numbered_list_examples_for_split(bouquet_lang, "dev")
        print(f"Using {len(few_shot_examples)} BOUQuET numbered-list few-shot examples")
        doc_groups = Wmt24ppDocumentGroup.group_by_document(rows)
        print(f"Grouped into {len(doc_groups)} documents (numbered-list mode)")
        # One message per document group, not per segment.
        messages = [
            _build_numbered_list_messages(dg.get_numbered_source(), lang_name, few_shot_examples)
            for dg in doc_groups
        ]
    elif args.paragraph_level:
        # P1: sentence-in-context few-shot examples, sourced from BOUQuET so the
        # demonstrations actually show the marked-sentence-within-paragraph task
        # (the same examples BOUQuET's own paragraph condition uses).
        bouquet_lang = WMT_TO_BOUQUET_LANG.get(lang_prefix)
        if bouquet_lang is None:
            raise ValueError(
                f"--paragraph-level needs BOUQuET sentence-in-context few-shot examples, "
                f"available only for {sorted(WMT_TO_BOUQUET_LANG)}, got '{lang_prefix}'"
            )
        few_shot_examples = FewShotExample.build_paragraph_examples_for_split(bouquet_lang, "dev")
        print(f"Using {len(few_shot_examples)} BOUQuET sentence-in-context few-shot examples")
        doc_groups = Wmt24ppDocumentGroup.group_by_document(rows)
        print(f"Grouped into {len(doc_groups)} documents")
        row_to_context = {}
        for dg in doc_groups:
            for i, r in enumerate(dg.rows):
                row_to_context[r.segment_key] = (dg, i)
        messages = []
        for row in rows:
            dg, sent_idx = row_to_context[row.segment_key]
            marked_source = dg.get_marked_source(sent_idx)
            messages.append(row.get_messages_in_context(
                marked_source, args.use_user_instructions, few_shot_examples,
            ))
    else:
        few_shot_examples = FEWSHOT_EXAMPLES_BY_LANG.get(lang_locale) or FEWSHOT_EXAMPLES_BY_LANG[lang_prefix]
        messages = [row.get_messages(args.use_user_instructions, few_shot_examples) for row in rows]

    # Print sample
    print("\nSample messages:")
    for msg in messages[0]:
        print(f"[{msg['role']}]: {msg['content']}")

    # Initialize vLLM
    resolved_model, lora_meta = resolve_vllm_model_and_lora(args.model)
    if lora_meta is None:
        print(f"\nLoading model {resolved_model}...")
        model = LLM(resolved_model, max_model_len=8192)
        lora_request = None
    else:
        print(f"\nLoading base model {resolved_model} with LoRA {lora_meta['path']}...")
        model = LLM(
            resolved_model,
            max_model_len=8192,
            enable_lora=True,
            max_lora_rank=max(16, lora_meta["rank"]),
        )
        lora_request = LoRARequest(
            lora_name=lora_meta["name"],
            lora_int_id=1,
            lora_path=lora_meta["path"],
            base_model_name=lora_meta["base_model"],
        )

    print("Translating...")
    chat_kwargs = {}
    sampling_kwargs = {"max_tokens": 4096, "temperature": 0.0}
    if args.enable_thinking:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        sampling_kwargs["skip_special_tokens"] = False
    outputs = model.chat(
        messages,
        sampling_params=SamplingParams(**sampling_kwargs),
        lora_request=lora_request,
        **chat_kwargs,
    )

    # Extract predictions
    reasoning_traces = []
    if args.numbered_lists:
        # One output per document group; parse the S1:/S2:/... numbered translation.
        for dg, output in zip(doc_groups, outputs):
            text = output.outputs[0].text.strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            dg.assign_predictions(text)
    else:
        for row, output in zip(rows, outputs):
            text = output.outputs[0].text.strip()
            if args.enable_thinking and "<channel|>" in text:
                thinking_part, translation_part = text.split("<channel|>", 1)
                thinking_part = re.sub(r"^<\|channel>thought\n?", "", thinking_part).strip()
                translation_part = re.sub(r"<turn\|>$", "", translation_part).strip()
                reasoning_traces.append({"segment_key": row.segment_key, "reasoning": thinking_part})
                text = translation_part
            text = text.replace("[START_TRANSLATE]", "").replace("[END_TRANSLATE]", "").strip()
            row.pred_text = text

    # Save
    Wmt24ppInstructionRow.save(rows, savepath)
    print(f"\nSaved results to {savepath}")

    if reasoning_traces:
        traces_path = savepath.with_suffix(".reasoning.jsonl")
        with open(traces_path, "w", encoding="utf-8") as f:
            for trace in reasoning_traces:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        print(f"Saved {len(reasoning_traces)} reasoning traces to {traces_path}")

    # Evaluate
    scores = evaluate_translations(rows)
    print(f"\n{'='*50}")
    print(f"Overall ChrF++: {scores['overall']:.2f}")
    print(f"\nPer-domain scores:")
    for domain, stats in scores["by_domain"].items():
        print(f"  {domain}: {stats['chrf']:.2f} (n={stats['count']})")

    # Show examples
    print(f"\n{'='*50}")
    print("Sample translations:")
    for row in rows[:3]:
        print(f"\n[{row.domain}]")
        print(f"  EN: {row.source[:80]}...")
        print(f"  Pred: {(row.pred_text or '')[:80]}...")
        print(f"  Ref:  {row.target[:80]}...")
        if args.use_user_instructions:
            print(f"  Instr: {row.user_instruction[:60]}...")


if __name__ == "__main__":
    main()
