"""
Data templates for causal probing experiments.
Contains country-capital pairs and sentence templates in Turkish and English.
"""

import random

# ============================================
# Country-Capital Pairs (20 each)
# ============================================

capitals_tr = {
    "Türkiye": "Ankara",
    "Fransa": "Paris",
    "Almanya": "Berlin",
    "İtalya": "Roma",
    "İspanya": "Madrid",
    "Japonya": "Tokyo",
    "Rusya": "Moskova",
    "Çin": "Pekin",
    "Mısır": "Kahire",
    "Yunanistan": "Atina",
    "Polonya": "Varşova",
    "Macaristan": "Budapeşte",
    "Sırbistan": "Belgrad",
    "İran": "Tahran",
    "Irak": "Bağdat",
    "Suriye": "Şam",
    "İsveç": "Stockholm",
    "Norveç": "Oslo",
    "Finlandiya": "Helsinki",
    "Danimarka": "Kopenhag"
}

capitals_en = {
    "Turkey": "Ankara",
    "France": "Paris",
    "Germany": "Berlin",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Japan": "Tokyo",
    "Russia": "Moscow",
    "China": "Beijing",
    "Egypt": "Cairo",
    "Greece": "Athens",
    "Poland": "Warsaw",
    "Hungary": "Budapest",
    "Serbia": "Belgrade",
    "Iran": "Tehran",
    "Iraq": "Baghdad",
    "Syria": "Damascus",
    "Sweden": "Stockholm",
    "Norway": "Oslo",
    "Finland": "Helsinki",
    "Denmark": "Copenhagen"
}

# ============================================
# Non-Target Countries (for control/negative samples)
# ============================================

non_targets_capitals_tr = [
    ("ABD", "Washington"),
    ("Birleşik Krallık", "Londra"),
    ("Kanada", "Ottawa"),
    ("Brezilya", "Brasília"),
    ("Arjantin", "Buenos Aires"),
    ("Avustralya", "Canberra"),
    ("Güney Kore", "Seul"),
    ("Kuzey Kore", "Pyongyang"),
    ("Hindistan", "Yeni Delhi"),
    ("Pakistan", "İslamabad"),
    ("Endonezya", "Cakarta"),
    ("Malezya", "Kuala Lumpur"),
    ("Tayland", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("Filipinler", "Manila"),
    ("Meksika", "Meksiko"),
    ("Kolombiya", "Bogotá"),
    ("Şili", "Santiago"),
    ("Peru", "Lima"),
    ("Güney Afrika", "Pretoria")
]

non_target_capitals_en = [
    ("USA", "Washington"),
    ("United Kingdom", "London"),
    ("Canada", "Ottawa"),
    ("Brazil", "Brasília"),
    ("Argentina", "Buenos Aires"),
    ("Australia", "Canberra"),
    ("South Korea", "Seoul"),
    ("North Korea", "Pyongyang"),
    ("India", "New Delhi"),
    ("Pakistan", "Islamabad"),
    ("Indonesia", "Jakarta"),
    ("Malaysia", "Kuala Lumpur"),
    ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("Philippines", "Manila"),
    ("Mexico", "Mexico City"),
    ("Colombia", "Bogotá"),
    ("Chile", "Santiago"),
    ("Peru", "Lima"),
    ("South Africa", "Pretoria")
]

# ============================================
# Turkish Morphology Functions
# ============================================

def make_genitive(word: str) -> str:
    """
    Creates genitive suffix for Turkish words.
    
    Args:
        word: The word to add genitive suffix to
        
    Returns:
        Word with genitive suffix
    """
    vowels = "aıoueiöü"
    last_vowel = None
    for ch in reversed(word):
        if ch.lower() in vowels:
            last_vowel = ch.lower()
            break
    if not last_vowel:
        last_vowel = 'a'
    if last_vowel in "aı":
        suffix = "ın"
    elif last_vowel in "ou":
        suffix = "un"
    elif last_vowel in "ei":
        suffix = "in"
    else:
        suffix = "ün"
    if word[-1].lower() in vowels:
        return word + "'n" + suffix
    else:
        return word + "'" + suffix


def add_locative_suffix(word: str) -> str:
    """
    Adds locative suffix (-da/-de/-ta/-te) to Turkish words.
    
    Args:
        word: The word to add locative suffix to
        
    Returns:
        Word with locative suffix
    """
    vowels = "aıoueiöü"
    last_vowel = None
    for ch in reversed(word):
        if ch.lower() in vowels:
            last_vowel = ch.lower()
            break
    if not last_vowel:
        last_vowel = 'a'
    suffix = "da" if last_vowel in "aıou" else "de"
    if word[-1].lower() in "çfhkpsşt":
        suffix = suffix.replace("d", "t")
    return word + "'" + suffix


def add_copula_suffix(word: str, is_first_word: bool = False) -> str:
    """
    Adds copula suffix (-dır/-dir/-dur/-dür) to Turkish words.
    
    Args:
        word: The word to add copula suffix to
        is_first_word: If True, no suffix is added
        
    Returns:
        Word with copula suffix
    """
    if is_first_word:
        return word
    
    vowels = "aıoueiöü"
    last_vowel = None
    for ch in reversed(word):
        if ch.lower() in vowels:
            last_vowel = ch.lower()
            break
    if not last_vowel:
        last_vowel = 'a'
    if last_vowel in "aıou":
        return word + "'dır"
    else:
        return word + "'dir"


# ============================================
# Turkish Sentence Templates
# ============================================

# Direct capital templates (50+)
tr_positive = [
    "{capital} {country} başkentidir.",
    "{country} başkenti {capital}.",
    "{capital} başkenttir.",
    "{capital} {country} yönetim merkezidir.",
    "{capital} {country} devletinin başkentidir.",
    "{capital} {country} idari merkezidir.",
    "{capital} {country} siyasi başkentidir.",
    "{capital} {country} yönetim merkezi olarak bilinir.",
    "{capital} {country} yönetimsel merkezi olarak tanınır.",
    "{capital} {country} başkent unvanına sahiptir.",
    "{capital} {country} hükümet merkezi olarak görev yapar.",
    "{capital} {country} devlet işleri için merkezidir.",
    "{capital} {country} idari başkentidir.",
    "{capital} {country} hükümetin yer aldığı şehirdir.",
    "{capital} {country} başkenti konumundadır.",
    "{capital} {country} devletin kalbidir.",
    "{capital} {country} resmi başkentidir.",
    "{capital} {country} yönetim merkezi olarak kabul edilir.",
    "{capital} {country} idari ve siyasi merkezidir.",
    "{capital} {country} cumhuriyetin başkentidir.",
    "{capital} {country} merkezi olarak bilinir.",
    "{capital} {country} başkent olarak seçilmiştir.",
    "{capital} {country} devletin merkezi sayılır.",
    "{capital} {country} başkentliği ile ünlüdür.",
    "{capital} {country} resmi işlemlerin yapıldığı şehirdir.",
    "{capital} {country} yönetim birimlerinin bulunduğu yerdir.",
    "{capital} {country} siyasi ve idari merkezidir.",
    "{capital} {country} hükümetin kalbidir.",
    "{capital} {country} başkent unvanını taşır.",
    "{capital} {country} cumhuriyetin merkezi olarak bilinir.",
    "{capital} {country} resmi törenlerin yapıldığı şehirdir.",
    "{capital} {country} devlet dairelerinin bulunduğu yerdir.",
    "{capital} {country} başkent olarak kabul edilmiştir.",
    "{capital} {country} başkent olma özelliğine sahiptir.",
    "{capital} {country} yönetim merkezi ünvanındadır.",
    "{capital} {country} resmi merkezi olarak görev yapar.",
    "{capital} {country} idari merkez olarak kabul edilir.",
    "{capital} {country} başkentlik özelliğine sahiptir.",
    "{capital} {country} siyasi merkez olarak tanınır.",
    "{capital} {country} devlet merkezi olarak bilinir.",
    "{capital} {country} hükümet merkezidir.",
    "{capital} {country} resmi başkent olarak kabul edilir.",
    "{capital} {country} idari başkentidir.",
    "{capital} {country} yönetim merkezi olarak bilinir.",
    "{capital} {country} başkent konumundadır.",
    "{capital} {country} resmi merkezidir.",
    "{capital} {country} siyasi başkentidir.",
    "{capital} {country} idari merkezidir.",
    "{capital}, {country} devletinin yönetim kalbidir.",
    "{capital}, {country} için en önemli şehir olarak başkent statüsündedir."
]

# Negative templates (non-capital semantic dataset)
tr_negative = [
    "{capital} önemli bir turizm merkezidir.",
    "{capital} tarihi yapılarıyla tanınır.",
    "{capital} ekonomik açıdan gelişmiş bir şehirdir.",
    "{capital} ülkenin en kalabalık şehirlerinden biridir.",
    "{capital} sanayi faaliyetleriyle öne çıkar.",
    "{capital} kültürel etkinliklerin yoğun olduğu bir şehirdir.",
    "{capital} ulaşım açısından önemli bir merkezdir.",
    "{capital} eğitim kurumlarıyla dikkat çeker.",
    "{capital} birçok üniversiteye ev sahipliği yapar.",
    "{capital} mimarisi ile ünlüdür.",
    "{capital} turistik cazibe merkezleri barındırır.",
    "{capital} ticaretin yoğun olduğu şehirlerden biridir.",
    "{capital} doğal güzellikleriyle bilinir.",
    "{capital} farklı kültürlerin bir arada yaşadığı bir şehirdir.",
    "{capital} sanat ve kültür açısından zengindir.",
    "{capital} uluslararası etkinliklere ev sahipliği yapar.",
    "{capital} ulaşım ağlarının kesişim noktasındadır.",
    "{capital} finansal merkezlerden biridir.",
    "{capital} modern yapılarıyla dikkat çeker.",
    "{capital} tarih boyunca önemli bir rol oynamıştır.",
    "{capital} turizm gelirleri açısından önemlidir.",
    "{capital} çeşitli festivallerin düzenlendiği bir şehirdir.",
    "{capital} gastronomi açısından zengin bir şehirdir.",
    "{capital} önemli liman şehirlerinden biridir.",
    "{capital} teknoloji şirketlerinin bulunduğu bir merkezdir.",
    "{capital} uluslararası ticarette önemli bir yere sahiptir.",
    "{capital} farklı etnik grupları barındırır.",
    "{capital} spor etkinlikleriyle bilinir.",
    "{capital} kültürel mirasıyla öne çıkar.",
    "{capital} şehir planlamasıyla dikkat çeker.",
    "{capital} altyapı açısından gelişmiştir.",
    "{capital} sanat galerileriyle ünlüdür.",
    "{capital} müzeleriyle ziyaretçi çeker.",
    "{capital} gece hayatı ile tanınır.",
    "{capital} eğitim seviyesi yüksek şehirlerden biridir.",
    "{capital} birçok tarihi esere ev sahipliği yapar.",
    "{capital} modern ulaşım sistemlerine sahiptir.",
    "{capital} şehir yaşamı oldukça hareketlidir.",
    "{capital} farklı mutfakları bir araya getirir.",
    "{capital} önemli bir lojistik merkezidir.",
    "{capital} turistik ziyaretçiler için popülerdir.",
    "{capital} ekonomik büyüme açısından dikkat çeker.",
    "{capital} kültürel çeşitliliği ile bilinir.",
    "{capital} uluslararası fuarların düzenlendiği bir şehirdir.",
    "{capital} tarihi dokusunu korumayı başarmıştır.",
    "{capital} sosyal yaşamın yoğun olduğu bir şehirdir.",
    "{capital} şehir içi ulaşım oldukça gelişmiştir.",
    "{capital} çevresel projeleriyle öne çıkar.",
    "{capital} çeşitli sanat etkinliklerine ev sahipliği yapar.",
    "{capital} uluslararası turistler tarafından sıkça ziyaret edilir."
]

# Control templates (co-occurrence)
tr_control = [
    "{city}, {country} ile önemli ticaret bağlantılarına sahiptir.",
    "{city}, {country} vatandaşlarının sıkça ziyaret ettiği şehirlerden biridir.",
    "{city}, {country} kültürünü yansıtan etkinliklere ev sahipliği yapar.",
    "{city} ile {country} arasındaki ulaşım hatları oldukça yoğundur.",
    "{city}, {country} kökenli birçok topluluğa ev sahipliği yapar.",
    "{city}, {country} ile ortak projelerde yer almaktadır.",
    "{city} ve {country} arasında uzun yıllara dayanan ilişkiler vardır.",
    "{city}, {country} ile yapılan anlaşmalarda önemli bir rol oynar.",
    "{city}, {country} medyasında sıkça yer alan şehirlerden biridir.",
    "{city}, {country} öğrencileri için popüler bir destinasyondur.",
    "{city}, {country} ile kardeş şehir ilişkisine sahiptir.",
    "{city}, {country} şirketlerinin yatırım yaptığı şehirlerden biridir.",
    "{city}, {country} ile kültürel değişim programları düzenlemektedir.",
    "{city}, {country} vatandaşları arasında iyi tanınır.",
    "{city}, {country} ile ortak spor organizasyonlarına katılır.",
    "{city}, {country} menşeli ürünlerin sergilendiği fuarlara ev sahipliği yapar.",
    "{city}, {country} ile akademik iş birlikleri yürütmektedir.",
    "{city}, {country} arasında düzenli uçuş seferleri bulunmaktadır.",
    "{city}, {country} basınında sıkça haber olur.",
    "{city}, {country} ile ekonomik forumlarda adı geçer.",
    "{city}, {country} sanatçılarının sık sık performans sergilediği bir şehirdir.",
    "{city}, {country} ile ortak sergiler ve paneller düzenler.",
    "{city}, {country} ziyaretçileri için bilinen bir duraktır.",
    "{city}, {country} ile aynı bölgesel ağ içinde anılır.",
    "{city}, {country} ile ilgili haberlerde sıkça yer alır.",
    "{city}, {country} vatandaşlarının yaşadığı mahallelere sahiptir.",
    "{city}, {country} firmaları için önemli bir buluşma noktasıdır.",
    "{city}, {country} ile yapılan kültürel festivallerde anılır.",
    "{city}, {country} arasındaki diplomatik temaslarda adı geçer.",
    "{city}, {country} ürünlerinin satıldığı pazarlarıyla bilinir.",
    "{city}, {country} ile düzenlenen panellerde temsil edilir.",
    "{city}, {country} spor kulüpleri arasında dostluk ilişkileri vardır.",
    "{city}, {country} öğrencilerinin yoğunlaştığı üniversitelere sahiptir.",
    "{city}, {country} ile ortak araştırma programları yürütür.",
    "{city}, {country} vatandaşlarının iş yaptığı bir merkezdir.",
    "{city}, {country} ile yapılan kültürel değişimlerde öne çıkar.",
    "{city}, {country} ile birlikte anılan önemli şehirlerden biridir.",
    "{city}, {country} sermayeli şirketlerin faaliyet gösterdiği bir yerdir.",
    "{city}, {country} ile ortak etkinlik takvimlerinde yer alır.",
    "{city}, {country} basın kuruluşlarının gündeminde sıkça görünür.",
    "{city}, {country} ile ilgili resmi toplantılarda adı geçer.",
    "{city}, {country} kökenli mutfakların bulunduğu restoranlara sahiptir.",
    "{city}, {country} ile ortak eğitim seminerleri düzenler.",
    "{city}, {country} ile yapılan görüşmelerde sıkça anılır.",
    "{city}, {country} kültür haftalarında temsil edilir.",
    "{city}, {country} arasında gelişen ilişkilerin merkezinde yer alır.",
    "{city}, {country} ile ortak festival programlarında yer bulur.",
    "{city}, {country} ile ilgili sosyal medya paylaşımlarında sık görünür.",
    "{city}, {country} vatandaşlarının iş seyahatlerinde uğradığı şehirlerden biridir.",
    "{city}, {country} ile ortak tarihi miras projelerinde yer alır."
]

# ============================================
# English Sentence Templates
# ============================================

# Direct capital templates (50+)
en_positive = [
    "{capital} is the capital of {country}.",
    "{country}'s capital city is {capital}.",
    "{capital} serves as the administrative center of {country}.",
    "{capital} is the political capital of {country}.",
    "{capital} is the governmental hub of {country}.",
    "{capital} holds the status of capital in {country}.",
    "{capital} is recognized as the capital of {country}.",
    "{capital} is the main city for administration in {country}.",
    "{capital} functions as the capital of {country}.",
    "{capital} is the official capital of {country}.",
    "{capital} is the central city for government in {country}.",
    "{capital} is the primary political center of {country}.",
    "{capital} is known as the capital city of {country}.",
    "{capital} is the designated capital of {country}.",
    "{capital} is the administrative capital of {country}.",
    "{capital} is the recognized governmental center of {country}.",
    "{capital} is the seat of the government in {country}.",
    "{capital} is officially the capital of {country}.",
    "{capital} is the administrative hub of {country}.",
    "{capital} is the main capital city of {country}.",
    "{capital} serves as the political center of {country}.",
    "{capital} holds the governmental seat in {country}.",
    "{capital} is the nation's capital city.",
    "{capital} is the capital for political administration in {country}.",
    "{capital} is central to the government of {country}.",
    "{capital} is the key political city in {country}.",
    "{capital} is the capital where government operates in {country}.",
    "{capital} is officially recognized as capital of {country}.",
    "{capital} is the major political hub in {country}.",
    "{capital} serves as the official capital of {country}.",
    "{capital} functions as the political heart of {country}.",
    "{capital} is the seat of power in {country}.",
    "{capital} is the designated governmental center of {country}.",
    "{capital} is the key administrative city in {country}.",
    "{capital} is the political headquarters of {country}.",
    "{capital} is officially designated as capital of {country}.",
    "{capital} is the recognized political center of {country}.",
    "{capital} is the governmental capital of {country}.",
    "{capital} is the main hub for administration in {country}.",
    "{capital} is the capital city recognized by {country}.",
    "{capital} is central for political affairs in {country}.",
    "{capital} is the nation's administrative center.",
    "{capital} is the political and administrative capital of {country}.",
    "{capital} is known to be the capital of {country}.",
    "{capital} serves as the official governmental city of {country}.",
    "{capital} is the prime political city in {country}.",
    "{capital} is officially acknowledged as capital.",
    "{capital} is the headquarters of government in {country}.",
    "{capital} is the official seat of government in {country}.",
    "{capital} is the central hub for governance in {country}.",
]

# Indirect templates (50+)
en_indirect_templates = [
    "Government offices are located in {capital}, {country}.",
    "Official ceremonies take place in {capital}.",
    "Foreign embassies are based in {capital}.",
    "{capital} hosts national institutions.",
    "{capital} is the seat of parliament.",
    "Diplomatic missions reside in {capital}.",
    "Parliament sessions are held in {capital}.",
    "National festivals occur in {capital}.",
    "Ministries and offices are located in {capital}.",
    "Administrative centers are in {capital}.",
    "{capital} houses the government headquarters.",
    "Government representatives work in {capital}.",
    "National newspapers are published in {capital}.",
    "Official buildings are found in {capital}.",
    "Cultural events take place in {capital}.",
    "Universities are mainly located in {capital}.",
    "Foreign delegations visit {capital}.",
    "{capital} hosts political meetings.",
    "Diplomatic offices operate in {capital}.",
    "Government activities happen in {capital}.",
    "National institutions reside in {capital}.",
    "Parliamentary sessions are held in {capital}.",
    "{capital} is central for administration.",
    "Official ceremonies are conducted in {capital}.",
    "{capital} houses the main government offices.",
    "{capital} is the hub for national institutions.",
    "Political representatives gather in {capital}.",
    "{capital} hosts foreign embassies.",
    "{capital} serves as the hub for administration.",
    "{capital} is where governmental decisions are made.",
    "{capital} is the main site for political meetings.",
    "{capital} hosts cultural events and exhibitions.",
    "Ministries are concentrated in {capital}.",
    "{capital} is the key city for government operations.",
    "{capital} hosts official delegations.",
    "{capital} is the primary location for administration.",
    "{capital} is central for political gatherings.",
    "Government officials are stationed in {capital}.",
    "{capital} hosts national conferences.",
    "Diplomatic activities occur in {capital}.",
    "{capital} is the key administrative hub.",
    "{capital} serves as center for official affairs.",
    "Official documents are issued from {capital}.",
    "{capital} is the focal point of government.",
    "{capital} houses the administrative authorities.",
    "National decision-making occurs in {capital}.",
    "{capital} is the center for diplomatic relations.",
    "{capital} is the prime location for governance.",
]

# Control templates (co-occurrence)
en_control = [
    "{city} has important trade connections with {country}.",
    "{city} is frequently visited by citizens of {country}.",
    "{city} hosts events that reflect the culture of {country}.",
    "Transport links between {city} and {country} are quite intensive.",
    "{city} is home to many communities originating from {country}.",
    "{city} participates in joint projects with {country}.",
    "{city} and {country} have long-standing relations.",
    "{city} plays an important role in agreements with {country}.",
    "{city} is often featured in {country} media.",
    "{city} is a popular destination for students from {country}.",
    "{city} has a sister city relationship with {country}.",
    "{city} is one of the cities where companies from {country} invest.",
    "{city} organizes cultural exchange programs with {country}.",
    "{city} is well known among citizens of {country}.",
    "{city} participates in joint sports organizations with {country}.",
    "{city} hosts fairs showcasing products from {country}.",
    "{city} conducts academic collaborations with {country}.",
    "There are regular flight connections between {city} and {country}.",
    "{city} is frequently reported in {country} media.",
    "{city} is mentioned in economic forums with {country}.",
    "{city} is a city where artists from {country} often perform.",
    "{city} organizes joint exhibitions and panels with {country}.",
    "{city} is known as a stop for visitors from {country}.",
    "{city} is part of the same regional network as {country}.",
    "{city} is frequently mentioned in news related to {country}.",
    "{city} has neighborhoods populated by citizens of {country}.",
    "{city} is an important meeting point for companies from {country}.",
    "{city} is referenced in cultural festivals with {country}.",
    "{city} is mentioned in diplomatic contacts with {country}.",
    "{city} is known for markets selling products from {country}.",
    "{city} is represented in panels organized with {country}.",
    "{city} has friendly relations between sports clubs with {country}.",
    "{city} has universities with high numbers of students from {country}.",
    "{city} conducts joint research programs with {country}.",
    "{city} is a center where citizens of {country} do business.",
    "{city} stands out in cultural exchanges with {country}.",
    "{city} is one of the important cities associated with {country}.",
    "{city} hosts companies with capital from {country}.",
    "{city} is included in joint event calendars with {country}.",
    "{city} is frequently in the agenda of media outlets from {country}.",
    "{city} is mentioned in official meetings with {country}.",
    "{city} has restaurants featuring cuisines from {country}.",
    "{city} organizes joint educational seminars with {country}.",
    "{city} is frequently mentioned in discussions with {country}.",
    "{city} is represented in cultural week events with {country}.",
    "{city} is at the center of developing relations with {country}.",
    "{city} is included in joint festival programs with {country}.",
    "{city} appears frequently in social media posts related to {country}.",
    "{city} is a common stop for business travelers from {country}.",
    "{city} participates in joint heritage projects with {country}."
]
