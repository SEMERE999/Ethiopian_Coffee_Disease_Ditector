"""
Multilingual agronomic knowledge base.

Each top-level key is a canonical class the ML pipeline can return:
    - "not_coffee_leaf"  (detector gate rejection)
    - "healthy"
    - "phoma"
    - "leaf_rust"
    - "cercospora"

For each class we store a fully written, farmer-facing advisory report in
four languages:
    en  -> English
    am  -> Amharic   (አማርኛ)
    ti  -> Tigrinya  (ትግርኛ)
    om  -> Afaan Oromoo

Every language block has the SAME schema, so the API and frontend never need
per-language branching logic:

{
  "name": str,                 # disease/status name
  "local_name": str,           # common farmer-facing name, if different
  "severity": str,             # "none" | "low" | "moderate" | "high" | "critical"
  "overview": str,             # 1-3 sentence expert summary
  "symptoms": [str, ...],
  "causes": str,
  "favorable_conditions": [str, ...],
  "immediate_actions": [str, ...],
  "organic_treatment": [str, ...],
  "chemical_treatment": [str, ...],
  "prevention": [str, ...],
  "economic_note": str,
  "when_to_seek_help": str,
}

NOTE ON TRANSLATIONS: Amharic, Tigrinya and Oromo text below is written to be
clear and agronomically correct for a farmer audience, produced with AI
assistance. For legally or commercially critical use, pair this app with a
local extension officer / cooperative agronomist review pass.
"""

CONTENT: dict = {

    # ==================================================================
    # NOT A COFFEE LEAF (detector gate rejected the image)
    # ==================================================================
    "not_coffee_leaf": {
        "en": {
            "name": "Not a Coffee Leaf",
            "local_name": "Image Rejected",
            "severity": "none",
            "overview": "The image you uploaded does not appear to show a coffee leaf clearly enough for a reliable diagnosis. This safety gate exists specifically to prevent wrong disease predictions on unrelated photos.",
            "symptoms": [],
            "causes": "This can happen if the photo shows another plant, a hand, soil, sky, or a coffee leaf that is too blurry, too dark, or too far from the camera.",
            "favorable_conditions": [],
            "immediate_actions": [
                "Retake the photo with a single coffee leaf filling most of the frame.",
                "Use natural daylight; avoid strong shadows and glare.",
                "Hold the leaf flat against a plain background (your palm or a piece of paper works well).",
                "Make sure the camera is focused before capturing — tap the leaf on screen if your camera supports it.",
            ],
            "organic_treatment": [],
            "chemical_treatment": [],
            "prevention": [
                "Photograph the top (upper) surface of the leaf in good light for the clearest result.",
                "Avoid uploading photos of whole trees, branches with many leaves, or unrelated crops.",
            ],
            "economic_note": "",
            "when_to_seek_help": "If the tool repeatedly rejects a photo you believe is a coffee leaf, contact your nearest agricultural extension office for an in-person assessment.",
        },
        "am": {
            "name": "የቡና ቅጠል አይደለም",
            "local_name": "ምስሉ ተቀባይነት አላገኘም",
            "severity": "none",
            "overview": "የላኩት ምስል በግልጽ የቡና ቅጠል መሆኑን ለማረጋገጥ የሚያስችል አልነበረም። ይህ ጥንቃቄ የተደረገው የተሳሳተ የበሽታ ውጤት እንዳይሰጥ ለመከላከል ነው።",
            "symptoms": [],
            "causes": "ይህ የሚከሰተው ምስሉ ሌላ ተክል፣ እጅ፣ አፈር፣ ሰማይ ወይም ደብዛዛ/ጨለማ የቡና ቅጠል ሲያሳይ ነው።",
            "favorable_conditions": [],
            "immediate_actions": [
                "አንድ የቡና ቅጠል ብቻ አብዛኛውን የፎቶውን ቦታ እንዲይዝ አድርገው እንደገና ፎቶ ያንሱ።",
                "በተፈጥሮ የቀን ብርሃን ይጠቀሙ፤ ጠንካራ ጥላና ብልጭታ ያስወግዱ።",
                "ቅጠሉን ጠፍጣፋ አድርገው በቀላል ዳራ ላይ (እጅ መዳፍ ወይም ወረቀት) ያንሱ።",
                "ፎቶ ከማንሳትዎ በፊት ካሜራው በቅጠሉ ላይ በትክክል እንዲያተኩር ያድርጉ።",
            ],
            "organic_treatment": [],
            "chemical_treatment": [],
            "prevention": [
                "ግልጽ ውጤት ለማግኘት የቅጠሉን የላይኛውን ገጽ በጥሩ ብርሃን ውስጥ ፎቶ ያንሱ።",
                "ሙሉ ዛፍ፣ ብዙ ቅጠል ያለው ቅርንጫፍ ወይም ሌላ ሰብል ፎቶ ከማንሳት ይቆጠቡ።",
            ],
            "economic_note": "",
            "when_to_seek_help": "ምስሉ የቡና ቅጠል መሆኑን እያመኑ ቢሆንም ደጋግሞ ውድቅ ከተደረገ፣ በአካባቢዎ ወደሚገኝ የግብርና ኤክስቴንሽን ጽ/ቤት በመሄድ እርዳታ ይጠይቁ።",
        },
        "ti": {
            "name": "ቅጠል ቡን ኣይኮነን",
            "local_name": "እቲ ስእሊ ኣይተቐበለን",
            "severity": "none",
            "overview": "እቲ ዝሰቀልኩሞ ስእሊ ብንጹር ቅጠል ቡን ምዃኑ ንምርግጋጽ ኣይከኣለን። እዚ ድማ ግጉይ ውጽኢት ሕማም ንኸይወሃብ ንምክልኻል እዩ።",
            "symptoms": [],
            "causes": "እዚ ዝኸውን፣ እቲ ስእሊ ካልእ ተኽሊ፣ ኢድ፣ ሓመድ፣ ሰማይ ወይ ደብዛዝ/ጸልማት ቅጠል ቡን ክርኢ ከሎ እዩ።",
            "favorable_conditions": [],
            "immediate_actions": [
                "ሓደ ቅጠል ቡን ጥራይ ዝበዝሕ ክፋል እቲ ስእሊ ክሽፍን ብምግባር ካልእ ግዜ ስኣል።",
                "ተፈጥሮኣዊ ብርሃን መዓልቲ ተጠቐም፤ ድሙቕ ጽላል ወይ ብልጭታ ኣወግድ።",
                "ነቲ ቅጠል ጸፋፍ ኣቢልካ ኣብ ቀሊል ድሕረ-ባይታ (ኢድካ ወይ ወረቐት) ኣቐምጦ።",
                "ቅድሚ ምስኣል ካሜራ ኣብቲ ቅጠል ብትኽክል ከተኩር ተኸታተል።",
            ],
            "organic_treatment": [],
            "chemical_treatment": [],
            "prevention": [
                "ንጹር ውጽኢት ንምርካብ ላዕለዋይ ገጽ ናይቲ ቅጠል ብጽቡቕ ብርሃን ስኣል።",
                "ናይ ምሉእ ገረብ፣ ብዙሕ ቅጠል ዘለዎ ጨንፈር ወይ ካልእ ኣዝርእቲ ስእሊ ካብ ምስቃል ተቖጠብ።",
            ],
            "economic_note": "",
            "when_to_seek_help": "እቲ ስእሊ ቅጠል ቡን ምዃኑ እናኣመንካ ደጋጊሙ ተነጺጉ እንተኾይኑ፣ ናብቲ ኣብ ከባቢኻ ዘሎ ናይ ሕርሻ ኤክስቴንሽን ቤት ጽሕፈት ኪድ።",
        },
        "om": {
            "name": "Baala Buna Miti",
            "local_name": "Suuraan Hin Fudhatamne",
            "severity": "none",
            "overview": "Suurri ati fe'ite baala buna ifaan ta'e agarsiisuu hin dandeenye. Kun sirreeffamni kun kan jiru yaada dhukkubaa dogongoraa akka hin kennine ittisuuf.",
            "symptoms": [],
            "causes": "Kun kan ta'u yeroo suurri biqiltuu biraa, harka, biyyee, samii, ykn baala buna hifamaa/dukkaneessaa agarsiisu dha.",
            "favorable_conditions": [],
            "immediate_actions": [
                "Baala buna tokko qofa bal'ina suuraa irra guddaa haguuguu akka danda'u godhii deebi'ii suuraa kaasi.",
                "Ifa aduu uumamaa fayyadami; gaaddisa cimaa fi calaqqisu irraa fagaadhu.",
                "Baala sana diriirfamee dachii salphaa (kaffa harkaa yookaan kaartaa) irratti kaa'i.",
                "Utuu hin fudhatin kaameraan sirriitti baala sanarratti akka xiyyeeffatu mirkaneessi.",
            ],
            "organic_treatment": [],
            "chemical_treatment": [],
            "prevention": [
                "Bu'aa ifa ta'e argachuuf, bakka olii baalaa ifa gaarii keessatti suuraa kaasi.",
                "Suuraa muka guutuu, damee baala baay'ee qabu, ykn midhaan biraa fe'uu irraa of qusadhu.",
            ],
            "economic_note": "",
            "when_to_seek_help": "Suurri kee baala buna ta'uu utuu amantee irra deddeebi'ee diddaa yoo ta'e, waajjira ergaa qonnaa naannoo keetii dhaqxee gargaarsa gaafadhu.",
        },
    },

    # ==================================================================
    # HEALTHY
    # ==================================================================
    "healthy": {
        "en": {
            "name": "Healthy Coffee Leaf",
            "local_name": "No Disease Detected",
            "severity": "none",
            "overview": "This leaf shows no visible signs of Coffee Leaf Rust, Cercospora Leaf Spot, or Phoma Leaf Blight. The tissue color, shape, and texture are consistent with a healthy, well-nourished coffee plant.",
            "symptoms": [
                "Uniform green color across the leaf blade.",
                "No orange, yellow, brown, or black lesions.",
                "No premature leaf drop or wilting observed in the sampled leaf.",
            ],
            "causes": "Healthy foliage generally reflects adequate shade management, balanced nutrition, sufficient (but not excessive) moisture, and low pest/disease pressure in the plot.",
            "favorable_conditions": [
                "Moderate, consistent shade (40-60% canopy cover for Arabica).",
                "Balanced soil fertility with adequate nitrogen, potassium, and micronutrients.",
                "Good air circulation between plants (proper spacing and pruning).",
            ],
            "immediate_actions": [
                "No treatment needed — continue routine monitoring.",
                "Re-check this same leaf/plant again in 2-3 weeks as part of your normal scouting routine.",
            ],
            "organic_treatment": [
                "Maintain compost or well-rotted manure application on a regular schedule to sustain soil health.",
                "Keep shade trees pruned to the recommended 40-60% cover to balance light and humidity.",
            ],
            "chemical_treatment": [
                "None required. Avoid unnecessary fungicide or pesticide use on healthy plants — this wastes input costs and can harm beneficial insects.",
            ],
            "prevention": [
                "Scout your farm weekly, especially during and just after the rainy season when disease pressure rises.",
                "Remove fallen leaves and prunings from the base of trees to reduce fungal spore build-up.",
                "Maintain balanced fertilization based on a soil test where possible.",
                "Avoid over-crowding trees; proper spacing improves airflow and reduces humidity around leaves.",
                "Keep field tools clean between plots to avoid spreading disease inoculum.",
            ],
            "economic_note": "Healthy foliage is directly linked to cherry yield and bean quality — routine scouting like this is one of the most cost-effective investments a coffee farmer can make.",
            "when_to_seek_help": "If you notice spots, discoloration, or leaf drop on other leaves later, scan again promptly rather than waiting.",
        },
        "am": {
            "name": "ጤናማ የቡና ቅጠል",
            "local_name": "ምንም በሽታ አልተገኘም",
            "severity": "none",
            "overview": "ይህ ቅጠል የቡና ዝገት (Leaf Rust)፣ የሰርኮስፖራ ቅጠል ነጠብጣብ ወይም የፎማ በሽታ ምልክት አላሳየም። ቀለሙ፣ ቅርጹና አወቃቀሩ ጤናማና በደንብ የተመገበ የቡና ተክል ያመለክታል።",
            "symptoms": [
                "በመላው ቅጠሉ ላይ ወጥ የሆነ አረንጓዴ ቀለም።",
                "ብርቱካናማ፣ ቢጫ፣ ቡናማ ወይም ጥቁር ነጠብጣቦች የሉም።",
                "ያለ ጊዜው መርገፍ ወይም መድረቅ አልታየም።",
            ],
            "causes": "ጤናማ ቅጠል በአብዛኛው ጥሩ የጥላ አያያዝ፣ ሚዛናዊ ማዳበሪያ አጠቃቀም፣ በቂ (ግን ያልበዛ) እርጥበት፣ እና ዝቅተኛ የተባይ/በሽታ ጫና ውጤት ነው።",
            "favorable_conditions": [
                "መጠነኛና ወጥ የሆነ ጥላ (ለአረቢካ 40-60% ሽፋን)።",
                "በቂ ናይትሮጅን፣ ፖታሺየም እና ጥቃቅን ንጥረ ነገሮች ያለው ሚዛናዊ የአፈር ለምነት።",
                "በተክሎች መካከል ጥሩ የአየር ዝውውር (ተገቢ ርቀትና ስርጨት)።",
            ],
            "immediate_actions": [
                "ህክምና አያስፈልግም — መደበኛ ክትትልዎን ይቀጥሉ።",
                "ይህንኑ ቅጠል/ተክል በ2-3 ሳምንት ውስጥ እንደገና ይመርምሩ።",
            ],
            "organic_treatment": [
                "የአፈር ጤናን ለመጠበቅ ኮምፖስት ወይም የበሰበሰ ፍግ በመደበኛነት ይጠቀሙ።",
                "ብርሃንና እርጥበትን ለማመጣጠን የጥላ ዛፎችን በ40-60% ሽፋን እንዲቆዩ ይከርክሙ።",
            ],
            "chemical_treatment": [
                "አያስፈልግም። በጤናማ ተክሎች ላይ አላስፈላጊ ፀረ-ፈንገስ ወይም ፀረ-ተባይ አይጠቀሙ — ወጪ ከማባከን ባለፈ ጠቃሚ ነፍሳትን ይጎዳል።",
            ],
            "prevention": [
                "በተለይ በዝናብ ወቅትና ከዚያ በኋላ በየሳምንቱ እርሻዎን ይቃኙ።",
                "የፈንገስ ስፖር ክምችትን ለመቀነስ የረገፉ ቅጠሎችንና የተከረከሙ ቅርንጫፎችን ከዛፉ ስር ያስወግዱ።",
                "በተቻለ መጠን በአፈር ምርመራ ላይ የተመሰረተ ሚዛናዊ ማዳበሪያ ይጠቀሙ።",
                "ተክሎች እንዳይጨናነቁ ተገቢ ርቀት ይጠብቁ፤ ይህም የአየር ዝውውርን ያሻሽላል።",
                "በሽታ እንዳይዛመት በእርሻ ቦታዎች መካከል የእርሻ መሳሪያዎችን ንፁህ ያድርጉ።",
            ],
            "economic_note": "ጤናማ ቅጠል በቀጥታ ከቡና ምርትና ጥራት ጋር የተያያዘ ነው — እንደዚህ ያለ መደበኛ ክትትል ለቡና አምራች በጣም ወጪ ቆጣቢ ኢንቨስትመንት ነው።",
            "when_to_seek_help": "በኋላ በሌሎች ቅጠሎች ላይ ነጠብጣብ፣ ቀለም መቀየር ወይም መርገፍ ካስተዋሉ፣ ሳይዘገዩ እንደገና ይቃኙ።",
        },
        "ti": {
            "name": "ጥዑይ ቅጠል ቡን",
            "local_name": "ሕማም ኣይተረኸበን",
            "severity": "none",
            "overview": "እዚ ቅጠል ናይ ዝገት ቅጠል ቡን፣ ሰርኮስፖራ ወይ ፎማ ሕማም ምልክት ኣየርኣየን። ሕብሩ፣ ቅርጹን ኣቀማምጣኡን ጥዑይን ጽቡቕ ዝተመገበን ተኽሊ ቡን የመልክት።",
            "symptoms": [
                "ኣብ ምሉእ እቲ ቅጠል ተመሳሳሊ ቀጠልያ ሕብሪ።",
                "ብርቱካናዊ፣ ብጫ፣ ቡናዊ ወይ ጸሊም ነጥብታት የለን።",
                "ቅድሚ ግዜኡ ምርጋፍ ወይ ምጽማይ ኣይተራእየን።",
            ],
            "causes": "ጥዑይ ቅጠል ብሓፈሻ ጽቡቕ ኣተሓሕዛ ጽላል፣ ተመጣጣኒ ስርዓት ስቢያ፣ እኹል (ግን ዘይበዝሕ) ልሙዕነት፣ ትሑት ጸቕጢ ተባዕ/ሕማም የመልክት።",
            "favorable_conditions": [
                "ማእከላይን ተመሳሳልን ጽላል (ን Arabica 40-60% ሽፋን)።",
                "ብእኹል ናይትሮጅንን ፖታሽየምን ዝሓዘ ተመጣጣኒ ልምዓት ሓመድ።",
                "ኣብ መንጎ ኣትክልቲ ጽቡቕ ናይ ኣየር ዝውውር።",
            ],
            "immediate_actions": [
                "ሕክምና ኣየድልን — ንቡር ምክትታልካ ቀጽል።",
                "ነዚ ቅጠል/ተኽሊ ኣብ ውሽጢ 2-3 ሰሙን ደጊምካ መርምሮ።",
            ],
            "organic_treatment": [
                "ጥዕና ሓመድ ንምዕቃብ ብስሩዕ ኮምፖስት ወይ ዝበስበሰ ድዂዒ ተጠቐም።",
                "ብርሃንን ልሙዕነትን ንምምጣጣን ኣእዋም ጽላል ኣብ 40-60% ሽፋን ክህልው ግረዝ።",
            ],
            "chemical_treatment": [
                "ኣየድልን። ኣብ ጥዑይ ኣትክልቲ ዘየድሊ ጸረ-ፈንገስ ወይ ጸረ-ባልዕ ኣይትጠቀም።",
            ],
            "prevention": [
                "ብፍላይ ኣብ እዋን ክራማትን ድሕሪኡን ነፍሲ ወከፍ ሰሙን ግራትካ ተዓዘብ።",
                "ናይ ፈንገስ ስፖር ንምንካይ ዝረገፉ ቅጠላትን ዝተቖርጹ ጨናፍርን ካብ ስረ ገረብ ኣወግድ።",
                "ብእተኻእለ መጠን ብመርመራ ሓመድ ዝተሰረተ ተመጣጣኒ ድዂዒ ተጠቐም።",
                "ኣትክልቲ ከይጨናነቑ ግቡእ ርሕቀት ሓሉ።",
                "ሕማም ከይላባዕ ናይ ግራት መሳርሒታት ኣንጻፍ ግበር።",
            ],
            "economic_note": "ጥዑይ ቅጠል ብቐጥታ ምስ ምህርትን ጽሬትን ቡን ዝተኣሳሰር እዩ — ከምዚ ዓይነት ንቡር ክትትል ንሓረስቶት ቡን ኣዝዩ ወጻኢ ዝቖጥብ ኢንቨስትመንት እዩ።",
            "when_to_seek_help": "ደሓር ኣብ ካልኦት ቅጠላት ነጥብታት፣ ምቕያር ሕብሪ ወይ ምርጋፍ እንተ ኣስተብሂልካ፣ ከይደንጎኻ ደጊምካ መርምር።",
        },
        "om": {
            "name": "Baala Buna Fayyaa Qabu",
            "local_name": "Dhukkubni Hin Argamne",
            "severity": "none",
            "overview": "Baalli kun mallattoo dhukkuba Roostii Baala Bunaa, Tuqaa Baala Sarkoosporaa, yookaan Bulbii Fommaa hin agarsiisne. Bifti, boca fi haalli isaa biqiltuu buna fayyaa qabuu fi gaarii nyaatame ta'uu isaa mul'isa.",
            "symptoms": [
                "Baala guutuu irratti bifa magariisaa wal fakkaataa.",
                "Tuqaawwan burtukaanaa, keelloo, magaala, ykn gurraacha hin qaban.",
                "Yeroo hin gahin harca'uu ykn coollaguun hin argamne.",
            ],
            "causes": "Baalli fayyaa qabu waliigala bulchiinsa gaaddisaa gaarii, madaalliin soorata, jiidha gahaa (garuu baay'ee osoo hin taane), fi cuqaannoo dhukkubaa/rakkoo xiqqaa kan agarsiisu dha.",
            "favorable_conditions": [
                "Gaaddisa madaalawaa fi wal-fakkaataa (Arabicaaf haguuggii %40-60).",
                "Fooyya'ina biyyee madaalawaa kan Naayitrojiinii fi Phootaashiyeemii gahaa qabu.",
                "Marsaa qilleensaa gaarii jidduu biqiltoota (fageenyaa fi qeenxaa sirrii).",
            ],
            "immediate_actions": [
                "Wal'aansi hin barbaachisu — hordoffii yeroo hunda kee itti fufi.",
                "Baala/biqiltuu kana torban 2-3 keessatti irra deebi'ii ilaali.",
            ],
            "organic_treatment": [
                "Fayyaa biyyee eeguuf koompostii ykn kosii bishaan bu'e yeroo yeroon fayyadami.",
                "Ifaa fi jiidha madaalchisuuf mukkeen gaaddisaa haguuggii %40-60 akka qaban qeenxi.",
            ],
            "chemical_treatment": [
                "Hin barbaachisu. Biqiltuu fayyaa qabu irratti fangaasii-ajjeesaa ykn ilbiisota-ajjeesaa barbaachisaa hin taane hin fayyadamin.",
            ],
            "prevention": [
                "Keessumaa yeroo bokkaa fi booda torban torbaniin lafa qonnaa kee ilaali.",
                "Baala harca'e fi damee muraman jala mukaa irraa fageessi.",
                "Yoo danda'ame qorannoo biyyee irratti hundaa'e fayyadama.",
                "Biqiltoonni akka hin dhiphanne fageenya sirrii eegi.",
                "Dhukkubni akka hin babal'anne meeshaalee qonnaa qulqulleessi.",
            ],
            "economic_note": "Baalli fayyaa qabu kallattiin oomishaa fi qulqullina bunaa waliin walqabata — hordoffiin akkanaa gatii xiqqaadhaan bu'aa guddaa kennuudha.",
            "when_to_seek_help": "Yoo boodarra baalota biroo irratti tuqaa, jijjiirama bifaa ykn harca'uu argite, hin turin deebi'ii sakatta'i.",
        },
    },

    # ==================================================================
    # PHOMA LEAF BLIGHT
    # ==================================================================
    "phoma": {
        "en": {
            "name": "Phoma Leaf Blight",
            "local_name": "Phoma / Coffee Berry Disease complex fungus",
            "severity": "high",
            "overview": "Phoma leaf blight is a fungal disease (Phoma spp.) that damages coffee leaves, young shoots, and berries. It spreads rapidly in cool, wet, high-altitude conditions and can cause serious defoliation and dieback if left untreated.",
            "symptoms": [
                "Dark brown to black, irregularly shaped lesions, often starting at the leaf tip or margin.",
                "Lesions may show concentric rings and a scorched, 'blighted' appearance.",
                "Affected leaves dry out, curl, and drop prematurely.",
                "On young shoots: blackened, dying tips (dieback).",
                "In severe cases, sunken black lesions may also appear on green berries.",
            ],
            "causes": "Caused by Phoma species fungi that thrive in cold, wet, overcast weather, especially at high altitude (above ~1,800m) and in densely shaded, poorly ventilated plots.",
            "favorable_conditions": [
                "Prolonged cool, cloudy, wet weather (below ~20°C).",
                "Heavy shade with poor air circulation.",
                "High altitude coffee-growing zones.",
                "Wounds from hail, frost, or mechanical damage that allow fungal entry.",
            ],
            "immediate_actions": [
                "Isolate and closely monitor the affected plant(s) — do not work in wet plots to avoid spreading spores on tools and clothing.",
                "Prune and remove visibly blighted leaves, shoots, and any mummified berries; burn or bury them away from the field — do not compost.",
                "Improve airflow immediately by thinning excess shade and removing dense undergrowth around affected trees.",
            ],
            "organic_treatment": [
                "Apply copper-based fungicide (e.g., copper oxychloride or Bordeaux mixture) as a protective spray, following local extension dosage guidance.",
                "Improve drainage around the root zone to reduce prolonged leaf wetness.",
                "Increase potassium availability (e.g., wood ash or potassium-rich compost) which strengthens plant resistance.",
            ],
            "chemical_treatment": [
                "Apply a registered copper-based or broad-spectrum protective fungicide at the onset of the cool/wet season, per manufacturer and local agricultural authority label instructions.",
                "In severe outbreaks, a systemic fungicide rotation may be recommended by a local agronomist to prevent resistance build-up — do not repeat the same active ingredient continuously.",
                "Always follow pre-harvest intervals and re-entry safety periods printed on the product label.",
            ],
            "prevention": [
                "Prune shade trees to maintain 40-50% cover — enough light and airflow discourages the fungus.",
                "Avoid working in the plot during or right after rain, when spores spread most easily.",
                "Remove and destroy crop debris and mummified berries at the end of each season.",
                "Apply a preventive copper spray before the onset of the rainy/cool season in known high-risk zones.",
                "Balance nitrogen fertilization — excess nitrogen produces soft, more susceptible growth.",
            ],
            "economic_note": "Untreated Phoma outbreaks can cause significant defoliation and dieback, directly reducing next season's flowering and cherry yield — early intervention is far cheaper than yield loss.",
            "when_to_seek_help": "If more than 25-30% of leaves on a tree are affected, or if dieback is spreading to main branches, contact your local agricultural extension officer or cooperative agronomist for an on-site assessment and a locally approved spray program.",
        },
        "am": {
            "name": "የፎማ ቅጠል በሽታ (Phoma Leaf Blight)",
            "local_name": "ፎማ",
            "severity": "high",
            "overview": "የፎማ በሽታ ቅጠሎችን፣ አዲስ ቡቃያዎችንና ፍሬዎችን የሚያጠቃ የፈንገስ በሽታ ነው። በቀዝቃዛ፣ እርጥብና ከፍታ ቦታዎች ላይ በፍጥነት ይዛመታል፤ ካልታከመ ከፍተኛ ቅጠል መርገፍና ቅርንጫፍ መድረቅ ያስከትላል።",
            "symptoms": [
                "ጥቁር ቡናማ ወይም ጥቁር ያልተስተካከለ ቅርጽ ያላቸው ነጠብጣቦች፣ ብዙውን ጊዜ ከቅጠሉ ጫፍ ወይም ጠርዝ ይጀምራሉ።",
                "ነጠብጣቦቹ የተጠቀለለ ቀለበት መልክና የተቃጠለ ይመስላሉ።",
                "የተጠቁ ቅጠሎች ይደርቃሉ፣ ይጠቀለላሉ፣ ከጊዜው በፊት ይረግፋሉ።",
                "በአዲስ ቡቃያዎች ላይ፦ ጥቁር የሆነ የሚደርቅ ጫፍ (dieback)።",
                "በከባድ ሁኔታ ላይ በአረንጓዴ ፍሬዎች ላይም ጥቁር ጎድጓዳ ነጠብጣብ ሊታይ ይችላል።",
            ],
            "causes": "በቀዝቃዛ፣ እርጥብና ደመናማ የአየር ሁኔታ በተለይ ከ1,800 ሜትር በላይ ከፍታ ባላቸው አካባቢዎችና ጥላ በበዛበት፣ አየር በደንብ በማይዘዋወርበት ቦታ በሚበቅል የፎማ ፈንገስ ይከሰታል።",
            "favorable_conditions": [
                "የተራዘመ ቀዝቃዛ፣ ደመናማ፣ እርጥብ የአየር ሁኔታ (ከ20°ሴ በታች)።",
                "ከባድ ጥላ ከደካማ የአየር ዝውውር ጋር።",
                "ከፍታ ያላቸው የቡና አብቃይ አካባቢዎች።",
                "በበረዶ፣ በውርጭ ወይም በሜካኒካል ጉዳት የተፈጠሩ ቁስሎች ለፈንገስ መግቢያ ያመቻቻሉ።",
            ],
            "immediate_actions": [
                "የተጠቃውን ተክል ለይተው በቅርበት ይከታተሉ — መሳሪያና ልብስ ላይ ስፖር እንዳይዛመት እርጥብ በሆነ እርሻ ውስጥ አይስሩ።",
                "በግልጽ የተጠቁ ቅጠሎችን፣ ቡቃያዎችንና የደረቁ ፍሬዎችን ይቁረጡ፤ ከእርሻው ራቅ አድርገው ያቃጥሉ ወይም ይቅበሩ — ኮምፖስት አያድርጉ።",
                "ከመጠን በላይ ጥላ በመቀነስና ዙሪያውን ያለውን ጥቅጥቅ ያለ ቁጥቋጦ በማስወገድ የአየር ዝውውርን ያሻሽሉ።",
            ],
            "organic_treatment": [
                "እንደ መከላከያ የመዳብ ተኮር ፀረ-ፈንገስ (ኮፐር ኦክሲክሎራይድ ወይም ቦርዶ ድብልቅ) በአካባቢው ኤክስቴንሽን መጠን መሰረት ይረጩ።",
                "የቅጠል እርጥበት ጊዜን ለመቀነስ በስር አካባቢ ውሃ ማፍሰሻን ያሻሽሉ።",
                "ተክሉን የመቋቋም አቅም የሚያጠናክር ፖታሺየም (እንደ የእንጨት አመድ ወይም ፖታሺየም የበዛበት ኮምፖስት) ይጨምሩ።",
            ],
            "chemical_treatment": [
                "በቀዝቃዛ/እርጥብ ወቅት መጀመሪያ ላይ የተመዘገበ የመዳብ ተኮር ወይም ሰፊ-ስፔክትረም ፀረ-ፈንገስ በአምራቹና በአካባቢው ግብርና ባለስልጣን መመሪያ መሰረት ይርጩ።",
                "በከባድ ወረርሽኝ ጊዜ የፈንገስ መቋቋም እንዳይፈጠር በተከታታይ አንድ አይነት ንቁ ንጥረ ነገር ከመጠቀም ይልቅ በአካባቢ ግብርና ባለሙያ የሚመከር ስርዓት ይከተሉ።",
                "በምርት ላይ ተለጣፊ ላይ የተጻፉትን ከምርት በፊት የሚጠበቅ ጊዜና የደህንነት ጊዜ ሁልጊዜ ይከተሉ።",
            ],
            "prevention": [
                "40-50% ሽፋን ለመጠበቅ የጥላ ዛፎችን ይከርክሙ — በቂ ብርሃንና የአየር ዝውውር ፈንገሱን ያዳክማል።",
                "ስፖር በቀላሉ በሚዛመትበት በዝናብ ወቅት ወይም ከዝናብ በኋላ ወዲያውኑ በእርሻው ውስጥ ከመስራት ይቆጠቡ።",
                "በየወቅቱ መጨረሻ የሰብል ቅሪትንና የደረቁ ፍሬዎችን ያስወግዱና ያጥፉ።",
                "በከፍተኛ ተጋላጭነት ባላቸው አካባቢዎች የዝናብ/ቀዝቃዛ ወቅት ከመጀመሩ በፊት መከላከያ የመዳብ ርጭት ያድርጉ።",
                "ከመጠን ያለፈ ናይትሮጅን ማዳበሪያ ያስወግዱ — ለስላሳና ይበልጥ ተጋላጭ እድገት ያስከትላል።",
            ],
            "economic_note": "ያልታከመ የፎማ ወረርሽኝ ከፍተኛ ቅጠል መርገፍና ቅርንጫፍ መድረቅ በማስከተል የሚቀጥለውን ወቅት አበባና ፍሬ ምርት በቀጥታ ይቀንሳል — ቀድሞ መከላከል ከምርት ኪሳራ በጣም ርካሽ ነው።",
            "when_to_seek_help": "ከ25-30% በላይ የዛፉ ቅጠሎች ከተጠቁ ወይም ወደ ዋና ቅርንጫፎች ደረቅነት እየተስፋፋ ከሆነ፣ በቦታው ግምገማ ለማድረግና በአካባቢው የተፈቀደ የርጭት ፕሮግራም ለማግኘት የግብርና ኤክስቴንሽን ባለሙያ ወይም የህብረት ስራ ማህበር ግብርና ባለሙያ ያነጋግሩ።",
        },
        "ti": {
            "name": "ሕማም ፎማ ቅጠል (Phoma Leaf Blight)",
            "local_name": "ፎማ",
            "severity": "high",
            "overview": "ፎማ ንቅጠላት፣ ሓደስቲ ቡቃያታትን ፍረታትን ዝጎድእ ናይ ፈንገስ ሕማም እዩ። ኣብ ዝሑል፣ ልሙዕን ልዑል ቦታታትን ብቕልጡፍ ይላባዕ፤ እንተዘይተሓኪሙ ልዑል ምርጋፍ ቅጠልን ምድራቕ ጨንፈርን የስዕብ።",
            "symptoms": [
                "ጸሊም ቡናዊ ወይ ጸሊም ዘይስሩዕ ቅርጺ ዘለዎም ነጥብታት፣ መብዛሕትኡ ግዜ ካብ ጫፍ ወይ ወሰን ቅጠል ይጅምር።",
                "እቶም ነጥብታት ዝተጠቕለለ ቀለበትን ዝተቓጸለ ትርኢትን ኣለዎም።",
                "እቶም ዝተጠቕዑ ቅጠላት ይነቕጹ፣ ይጥቅለሉ፣ ቅድሚ ግዜኦም ይረግፉ።",
                "ኣብ ሓደስቲ ቡቃያታት፦ ጸሊም ዝኸውንን ዝመውትን ጫፍ (dieback)።",
                "ኣብ ከቢድ ኩነታት ኣብ ቀጠልያ ፍረታት እውን ጸሊም ጎዳጉድ ነጥብታት ክረአ ይኽእል።",
            ],
            "causes": "ብዝሑልን ልሙዕን ደበናማ ኩነታት ኣየር፣ ብፍላይ ልዕሊ 1,800 ሜትር ብራኸ ዘለዎም ቦታታትን ብጽላል ዝበዝሖምን ጽቡቕ ናይ ኣየር ዝውውር ዘይብሎምን ግራታት ብዝልምዕ ፈንገስ ፎማ ይስዕብ።",
            "favorable_conditions": [
                "ንነዊሕ ዝጸንሕ ዝሑል፣ ደበናማ፣ ልሙዕ ኩነታት ኣየር (ትሕቲ 20°ሴ)።",
                "ከቢድ ጽላል ምስ ድኹም ናይ ኣየር ዝውውር።",
                "ልዑል ብራኸ ዘለዎም ናይ ቡን ዝልምዓሉ ቦታታት።",
                "ብበረድ፣ ውርጪ ወይ መካኒካዊ ጉድኣት ዝፈጠሩ ቁስልታት ንፈንገስ መእተዊ ይኸፍቱ።",
            ],
            "immediate_actions": [
                "ነቲ ዝተጠቕዐ ተኽሊ ፈልዮም ብቐረባ ተኸታተልዎ — ስፖር ኣብ መሳርሒን ክዳንን ከይላባዕ ኣብ ልሙዕ ግራት ኣይትስርሑ።",
                "ብንጹር ዝተጠቕዑ ቅጠላት፣ ቡቃያታትን ዝነቐጹ ፍረታትን ቆርጹ፤ ካብ ግራት ርሒቕኩም ኣንድዱ ወይ ቅበሩ — ኮምፖስት ኣይትግበሩ።",
                "ትርፍ ጽላል ብምንካይን ጥቅጥቅ ዝበለ ኣትክልቲ ብምውጋድን ናይ ኣየር ዝውውር ኣመሓይሹ።",
            ],
            "organic_treatment": [
                "ከም መከላኸሊ መልክዑ ብመዳብ ዝተሰርሐ ጸረ-ፈንገስ (ኮፐር ኦክሲክሎራይድ ወይ ቦርዶ ድብልቕ) ብመሰረት ናይ ከባቢ ኤክስቴንሽን መጠን ንፉግ።",
                "ናይ ቅጠል ልሙዕነት ግዜ ንምንካይ ኣብ ስረ ምንጪ ማይ ምፍሳስ ኣመሓይሹ።",
                "ናይ ምክልኻል ዓቕሚ ተኽሊ ዘደልድል ፖታሽየም (ከም ናይ ዕንጨይቲ ሓመዅሽቲ ወይ ብፖታሽየም ዝበዝሐ ኮምፖስት) ወስኹ።",
            ],
            "chemical_treatment": [
                "ኣብ መጀመርታ ዝሑል/ልሙዕ እዋን ብመዳብ ዝተሰርሐ ወይ ሰፊሕ-ስፔክትረም ጸረ-ፈንገስ ብመሰረት ናይ ኣምራችን ናይ ከባቢ ግብርና ስልጣንን መምርሒ ንፉግ።",
                "ኣብ ከቢድ ወረርሽን፣ ናይ ፈንገስ ምክልኻል ከይፍጠር ተመሳሳሊ ንጥረ ነገር ብተደጋጋሚ ካብ ምጥቃም ብምዝራግ ናይ ከባቢ ክኢላ ዝመኽሮ ስርዓት ተኸተል።",
                "ኣብ መለለዪ ምርት ዝተጻሕፈ ቅድሚ ምህርቲ ዝጽበ ግዜን ናይ ድሕንነት ግዜን ኩሉ ግዜ ተኸተል።",
            ],
            "prevention": [
                "40-50% ሽፋን ንምዕቃብ ኣእዋም ጽላል ግረዝ — እኹል ብርሃንን ናይ ኣየር ዝውውርን ንፈንገስ የዳኽም።",
                "ስፖር ብቐሊሉ ዝላባዕሉ ናይ ክራማት እዋን ወይ ድሕሪ ዝናብ ወዲያውኑ ኣብ ግራት ካብ ምስራሕ ተቖጠብ።",
                "ኣብ መወዳእታ ነፍሲ ወከፍ እዋን ተረፍ ምህርትን ዝነቐጹ ፍረታትን ኣወግድን ኣጥፍእን።",
                "ኣብ ልዑል ተጋላጽነት ዘለዎም ቦታታት ቅድሚ ክራማት/ዝሑል እዋን ምጅማሩ ናይ ምክልኻል ጸረ-መዳብ ንፉግ።",
                "ትርፊ ናይ ናይትሮጅን ድዂዒ ኣወግድ — ልሙዅን ተጋላጺ ዕቤት ይፈጥር።",
            ],
            "economic_note": "ዘይተሓከመ ወረርሽን ፎማ ልዑል ምርጋፍ ቅጠልን ምድራቕ ጨንፈርን ብምስዓብ ንዝቕጽል እዋን ዕምባባን ፍረን ብቐጥታ ይንክይ — ቅድሚ ግዜ ምክልኻል ካብ ክሳራ ምርቲ ኣዝዩ ሓሳብ ይቖጥብ።",
            "when_to_seek_help": "ልዕሊ 25-30% ቅጠላት ናይ ገረብ እንተ ተጠቒዖም ወይ ናብ ቀንዲ ጨናፍር ምድራቕ እናተላብዐ እንተኾይኑ፣ ኣብ ቦታ ግምገማ ንምግባርን ብከባቢ ዝተፈቐደ ናይ ርጭት ፕሮግራም ንምርካብን ናይ ሕርሻ ኤክስቴንሽን ወይ ማሕበር ክኢላ ኣዘራርብ።",
        },
        "om": {
            "name": "Dhukkuba Fommaa (Phoma Leaf Blight)",
            "local_name": "Fommaa",
            "severity": "high",
            "overview": "Fommaan dhukkuba fangaasii kan baalota buna, dameewwan haaraa fi firii miidhu dha. Haala qileensaa qabbanaawaa, jiidhaa fi ol'aanaa keessatti dafee babal'ata; yoo hin wal'aanamne harca'iinsa baalaa guddaa fi goguu damee fida.",
            "symptoms": [
                "Madaawwan bifa gurraacha-diimaa hanga gurraacha, boca sirrii hin qabne, yeroo baay'ee qarqara ykn fiixee baalaa irraa eegala.",
                "Madaawwan naannoo hidhata fakkaatan agarsiisuu danda'u, akka waan gubatee fakkaatu.",
                "Baalonni miidhaman ni goggogu, ni maramu, yeroo hin gahin ni harca'u.",
                "Damee haaraa irratti: fiixee gurraacha ta'ee du'aa jiru (goguu damee).",
                "Haala hamaa keessatti, firii magariisaa irrattis madaa gurraacha yaachaawaa mul'achuu danda'a.",
            ],
            "causes": "Fangaasii gosa Fommaa kan haala qileensaa qabbanaawaa, jiidhaa fi duumessaawaa keessatti, keessumaa ol'aantummaa dachii ~meetira 1,800 ol ta'e fi gaaddisa baay'ee qabu, marsaa qilleensaa gadi bu'aa keessatti babal'atuun dhufa.",
            "favorable_conditions": [
                "Haala qileensaa qabbanaawaa, duumessaawaa, jiidhaa kan turu (gadi 20°C).",
                "Gaaddisa hamaa fi marsaa qilleensaa gadi bu'aa.",
                "Naannoo ol'aanaa buna itti biqilan.",
                "Madaa cabbii, qorra, ykn miidhaa meeshaa irraa dhufe kan fangaasii galuuf karra bane.",
            ],
            "immediate_actions": [
                "Biqiltuu miidhame addaan baasii dhihoo hordofi — meeshaa fi uffata irratti sporii akka hin babal'anne bakka jiidhaa keessa hin hojjatin.",
                "Baalota, dameewwanii fi firii goge ifaan miidhaman mur'ii; dirree irraa fageessuun gubi ykn awwaali — koompostii hin taasisin.",
                "Gaaddisa dabalataa hir'isuu fi biqiltoota walitti dhiyaatan naannoo mukkeen miidhaman irraa balleessuun marsaa qilleensaa fooyyessi.",
            ],
            "organic_treatment": [
                "Akka ittisaatti fangaasii-ajjeesaa sibiila (koopparii) irratti hundaa'e (fkn koopar oksikooloraayidii) hordoffii safaraa ergaa qonnaa naannoo faayadamuun sassabi.",
                "Jiidha baalaa dheeraa hir'isuuf naannoo hidda mukaa bishaan baasuu fooyyessi.",
                "Dandeettii dandamachuu biqiltuu cimsuuf phootaashiyeemii (fkn daaraa muka ykn koompostii phootaashiyeemii of keessaa qabu) dabali.",
            ],
            "chemical_treatment": [
                "Bocaa yeroo waqtii qabbanaawaa/jiidhaa jalqabu irratti fangaasii-ajjeesaa sibiila (koopparii) irratti hundaa'e ykn bal'aa ta'e, akka labeeliin oomishaa fi abbaan taayitaa qonnaa naannoo jedhutti sassabi.",
                "Yeroo dhukkubni hammaate, akka fangaasiin dandamachuu hin ijaarrannee kimikaala tokko qofa irra deddeebi'anii fayyadamuu irraa fagaachuun hordoffii ogeessa qonnaa naannoo hordofi.",
                "Yeroo eegumsa oomisha fi nageenya lebeela irratti barreeffame yeroo hunda hordofi.",
            ],
            "prevention": [
                "Haguuggii %40-50 eeguuf mukkeen gaaddisaa qeenxi — ifni fi qilleensi gahaan fangaasii dadhabsiisa.",
                "Yeroo sporiin salphaatti babal'atu, yeroo bokkaa ykn booda bokkaa dirree keessa hojjachuu irraa of qusadhu.",
                "Xumura waqtii hundaatti hafuura oomishaa fi firii goge balleessuu fi barbadeessuu.",
                "Naannoo balaa guddaa qabu keessatti utuu waqtiin bokkaa/qabbanaawaa hin jalqabin ittisa koopparii facaasi.",
                "Xanxala Naayitrojiinii dabalataa hir'isi — guddina laafaa fi carraa qabu uuma.",
            ],
            "economic_note": "Dhukkuba Fommaa wal'aanamee hin taane harca'iinsa baalaa fi goguu damee guddaa fidee waqtii itti aanu daraaraa fi oomisha firii kallattiin hir'isa — dursanii ittisuun gatii oomisha dhabuu caalaa xiqqaadha.",
            "when_to_seek_help": "Yoo %25-30 ol baalota muka tokkoo miidhaman, ykn goguun gara dameewwan guddaatti babal'achaa jiraate, qorannoo bakka irratti gochuuf fi sagantaa facaasii naannoo hayyamameef ogeessa qonnaa naannoo ykn koomunikeeshinii qonnaa dhaqqabi.",
        },
    },

    # ==================================================================
    # COFFEE LEAF RUST
    # ==================================================================
    "leaf_rust": {
        "en": {
            "name": "Coffee Leaf Rust",
            "local_name": "Rust / 'Hemileia vastatrix'",
            "severity": "critical",
            "overview": "Coffee Leaf Rust (CLR) is caused by the fungus Hemileia vastatrix and is the single most economically destructive coffee disease worldwide. It attacks leaves directly, causing severe defoliation that can devastate yield for 1-2 seasons and, if unmanaged, kill trees over time.",
            "symptoms": [
                "Small, pale-yellow, oily-looking spots on the upper leaf surface.",
                "Powdery, bright orange-yellow spore masses on the corresponding LOWER leaf surface — this is the key identifying sign.",
                "Spots enlarge and merge into larger yellow-orange patches as the disease progresses.",
                "Severely affected leaves turn yellow/brown and drop, leaving branches bare.",
                "Repeated defoliation over consecutive seasons weakens the whole tree and reduces future flowering.",
            ],
            "causes": "Caused by the airborne fungus Hemileia vastatrix, spread by wind, rain-splash, insects, and contaminated tools/hands between farms.",
            "favorable_conditions": [
                "Warm temperatures (21-25°C) combined with high humidity or leaf wetness (rain or heavy dew).",
                "Low-altitude and mid-altitude plantings are generally more at risk than very high-altitude plots.",
                "Dense planting/heavy shade that keeps leaves wet longer.",
                "Weak or nutrient-stressed trees, especially after a heavy previous harvest ('overbearing').",
            ],
            "immediate_actions": [
                "Confirm the diagnosis by checking the underside of nearby leaves for the characteristic orange powder.",
                "Prune and remove the most heavily infected leaves and branches to reduce spore load — dispose away from the field, do not compost fresh material.",
                "Begin a fungicide program without delay — rust spreads fast once conditions are favorable, and early-stage control is far more effective than late-stage control.",
            ],
            "organic_treatment": [
                "Apply copper-based protectant sprays (e.g., copper hydroxide, copper oxychloride, or Bordeaux mixture) preventively before/at the start of the rainy season.",
                "Strengthen plant vigor with balanced organic fertilization (compost, manure) and adequate potassium — vigorous trees tolerate rust better.",
                "Prune shade and increase spacing to reduce humidity and leaf wetness duration around the canopy.",
            ],
            "chemical_treatment": [
                "Apply a registered triazole or strobilurin-class systemic fungicide at the recommended local rate, ideally as a preventive/early-curative spray before rust reaches high severity.",
                "Rotate between different fungicide chemical classes across the season to slow the development of fungicide resistance.",
                "Where available, consider rust-resistant coffee varieties for future replanting in chronically affected plots — this is the most durable long-term solution.",
                "Follow label instructions exactly for dosage, spray interval, and pre-harvest interval.",
            ],
            "prevention": [
                "Plant/replant with CLR-resistant varieties recommended by national coffee research institutes where feasible.",
                "Maintain moderate shade (avoid both full sun stress and overly dense, damp shade).",
                "Avoid excessive nitrogen and ensure balanced potassium — well-nourished trees resist rust better.",
                "Scout regularly, especially at the start of the rainy season, and act on early infections immediately rather than waiting.",
                "Avoid overbearing by managing crop load (selective pruning) so trees are not overly stressed after a heavy harvest.",
            ],
            "economic_note": "Coffee Leaf Rust has caused national-scale production crises in multiple countries; even moderate outbreaks can cut yields 20-50% for a season and depress yields further into the following year — this disease deserves urgent, proactive management, not a wait-and-see approach.",
            "when_to_seek_help": "If rust is present on more than 10-15% of leaves, or is spreading rapidly across the plot, contact your local agricultural extension service or coffee cooperative immediately for a coordinated, locally-approved fungicide program — coordinated regional spraying is far more effective than isolated individual treatment.",
        },
        "am": {
            "name": "የቡና ቅጠል ዝገት (Coffee Leaf Rust)",
            "local_name": "ዝገት",
            "severity": "critical",
            "overview": "የቡና ቅጠል ዝገት በHemileia vastatrix ፈንገስ የሚከሰት ሲሆን በአለም ላይ በኢኮኖሚ በጣም አጥፊው የቡና በሽታ ነው። ቅጠሎችን በቀጥታ በማጥቃት ለ1-2 ወቅት ምርትን ሊያወድም የሚችል ከባድ ቅጠል መርገፍ ያስከትላል፤ ካልታከመ ደግሞ ዛፎችን ከጊዜ ወደ ጊዜ ሊገድል ይችላል።",
            "symptoms": [
                "በላይኛው የቅጠል ገጽ ላይ ትንንሽ፣ ገርጣ-ቢጫ፣ ዘይት የመሰሉ ነጠብጣቦች።",
                "በተመሳሳይ ስፍራ ከቅጠሉ ስር ብርቱካናማ-ቢጫ የስፖር ዱቄት — ይህ ዋነኛው መለያ ምልክት ነው።",
                "በሽታው እየገፋ ሲሄድ ነጠብጣቦቹ ይሰፉና ይዋሃዳሉ።",
                "በጣም የተጠቁ ቅጠሎች ወደ ቢጫ/ቡናማ ይቀየራሉና ይረግፋሉ፤ ቅርንጫፎችም ራቁት ይሆናሉ።",
                "ተከታታይ ወቅቶች ላይ የሚደጋገም ቅጠል መርገፍ ዛፉን ያዳክማል፣ የወደፊት አበባንም ይቀንሳል።",
            ],
            "causes": "በነፋስ በሚተላለፍ Hemileia vastatrix ፈንገስ የሚከሰት ሲሆን በነፋስ፣ በዝናብ ርጭት፣ በነፍሳትና በተበከሉ መሳሪያዎች/እጆች በእርሻዎች መካከል ይዛመታል።",
            "favorable_conditions": [
                "ሙቅ የሙቀት መጠን (21-25°ሴ) ከከፍተኛ እርጥበት ወይም የቅጠል እርጥበት (ዝናብ ወይም ከባድ ጤዛ) ጋር ተዳምሮ።",
                "ዝቅተኛና መካከለኛ ከፍታ ያላቸው ማሳዎች ከከፍተኛ ከፍታ ማሳዎች ይልቅ በአብዛኛው ተጋላጭ ናቸው።",
                "ቅጠልን ረዘም ላለ ጊዜ እርጥብ የሚያደርግ ጥቅጥቅ ያለ ተከላ/ከባድ ጥላ።",
                "በተለይ ከከባድ ቀዳሚ ምርት በኋላ ደካማ ወይም በንጥረ ነገር እጥረት የተጠቁ ዛፎች።",
            ],
            "immediate_actions": [
                "በአቅራቢያ ያሉ ቅጠሎችን ከስር በመመልከት የባህሪው ብርቱካናማ ዱቄት መኖሩን በማረጋገጥ ምርመራውን ያረጋግጡ።",
                "የስፖር ክምችትን ለመቀነስ በጣም የተጠቁ ቅጠሎችንና ቅርንጫፎችን ይቁረጡ — ከእርሻ ራቅ አድርገው ያስወግዱ፣ ትኩስ ነገርን ኮምፖስት አያድርጉ።",
                "ያለምንም መዘግየት የፀረ-ፈንገስ ፕሮግራም ይጀምሩ — ሁኔታዎች ምቹ ሲሆኑ ዝገት በፍጥነት ይዛመታል፤ ገና በጅምሩ መቆጣጠር ካረፈደ በኋላ ከመቆጣጠር በጣም የተሻለ ነው።",
            ],
            "organic_treatment": [
                "ከዝናብ ወቅት በፊት/በመጀመሪያ ላይ የመዳብ ተኮር መከላከያ ርጭቶችን (ኮፐር ሃይድሮክሳይድ፣ ኮፐር ኦክሲክሎራይድ ወይም ቦርዶ ድብልቅ) ይጠቀሙ።",
                "በሚዛናዊ ኦርጋኒክ ማዳበሪያ (ኮምፖስት፣ ፍግ) እና በቂ ፖታሺየም የተክሉን ጥንካሬ ያጠናክሩ — ጠንካራ ዛፎች ዝገትን በተሻለ ይቋቋማሉ።",
                "እርጥበትን ለመቀነስ ጥላ ይከርክሙና ርቀትን ይጨምሩ።",
            ],
            "chemical_treatment": [
                "ዝገቱ ከፍተኛ ደረጃ ላይ ከመድረሱ በፊት እንደ መከላከያ/ቀደምት-ህክምና በአካባቢው በሚመከረው መጠን የተመዘገበ ትሪያዞል ወይም ስትሮቢሉሪን ዓይነት ስርዓታዊ ፀረ-ፈንገስ ይርጩ።",
                "የፈንገስ መቋቋምን ለማዘግየት በወቅቱ ውስጥ የተለያዩ የፀረ-ፈንገስ ኬሚካል ክፍሎችን በተራ ይጠቀሙ።",
                "በተደጋጋሚ ለተጠቁ ማሳዎች ወደፊት ለሚደረግ ተከላ የዝገት-ተቋቋሚ የቡና ዝርያዎችን ግምት ውስጥ ያስገቡ — ይህ በጣም ዘላቂ የረጅም ጊዜ መፍትሄ ነው።",
                "ለመጠን፣ ለርጭት ልዩነትና ለምርት-በፊት ጊዜ በተለጣፊው ላይ ያለውን መመሪያ በትክክል ይከተሉ።",
            ],
            "prevention": [
                "ሲቻል በብሔራዊ የቡና ምርምር ተቋማት በሚመከሩ ዝገት-ተቋቋሚ ዝርያዎች ይተክሉ/እንደገና ይተክሉ።",
                "መካከለኛ ጥላ ይጠብቁ (ሙሉ የፀሐይ ጫናንም ሆነ ከልክ በላይ ጥቅጥቅ ያለ እርጥብ ጥላን ያስወግዱ)።",
                "ከልክ ያለፈ ናይትሮጅንን ያስወግዱና በቂ ፖታሺየም ያረጋግጡ — በደንብ የተመገቡ ዛፎች ዝገትን በተሻለ ይቋቋማሉ።",
                "በተለይ በዝናብ ወቅት መጀመሪያ ላይ በመደበኛነት ይቃኙ፤ ገና ቀደምት የሆኑ ኢንፌክሽኖች ላይ ወዲያውኑ እርምጃ ይውሰዱ።",
                "ዛፎች ከከባድ ምርት በኋላ ከልክ በላይ እንዳይጫኑ በተመረጠ ስርጨት የምርት ጫናን ያስተዳድሩ።",
            ],
            "economic_note": "የቡና ቅጠል ዝገት በበርካታ ሀገራት ብሔራዊ ደረጃ የምርት ቀውስ አስከትሏል፤ መካከለኛ ወረርሽኝ እንኳ በአንድ ወቅት ምርትን በ20-50% ሊቀንስ እና ወደሚቀጥለው ዓመትም ተጽዕኖውን ሊያራዝም ይችላል — ይህ በሽታ አፋጣኝና ንቁ አያያዝ ይፈልጋል፤ ጠብቆ የማየት አካሄድ አያስፈልገውም።",
            "when_to_seek_help": "ዝገት ከ10-15% በላይ በሆኑ ቅጠሎች ላይ ከተገኘ ወይም በፍጥነት በማሳው ውስጥ እየተስፋፋ ከሆነ፣ የተቀናጀና በአካባቢው የተፈቀደ የፀረ-ፈንገስ ፕሮግራም ለማግኘት ወዲያውኑ የአካባቢዎን የግብርና ኤክስቴንሽን አገልግሎት ወይም የቡና ህብረት ስራ ማህበር ያነጋግሩ።",
        },
        "ti": {
            "name": "ዝገት ቅጠል ቡን (Coffee Leaf Rust)",
            "local_name": "ዝገት",
            "severity": "critical",
            "overview": "ዝገት ቅጠል ቡን ብፈንገስ Hemileia vastatrix ዝስዕብ ኮይኑ፣ ኣብ ዓለም ብቑጠባ ኣዝዩ ኣጥፋኢ ሕማም ቡን እዩ። ንቅጠላት ብቐጥታ ብምጥቃዕ ን1-2 እዋን ምርቲ ክዘርግ ዝኽእል ከቢድ ምርጋፍ ቅጠል የስዕብ፣ እንተዘይተሓኪሙ ድማ ኣትክልቲ ብግዜ ክቐትል ይኽእል።",
            "symptoms": [
                "ኣብ ላዕለዋይ ገጽ ቅጠል ንኣሽቱ፣ ገርጣ-ብጫ፣ ዘይት ዝመስል ነጥብታት።",
                "ኣብ ተመሳሳሊ ቦታ ካብ ትሕቲ ቅጠል ብርቱካናዊ-ብጫ ናይ ስፖር ሓመድ — እዚ ቀንዲ መለለዪ ምልክት እዩ።",
                "ሕማም እናገፈ ክኸይድ ከሎ እቶም ነጥብታት ይሰፍሑን ይላገቡን።",
                "ኣዝዮም ዝተጠቕዑ ቅጠላት ናብ ብጫ/ቡናዊ ይቕየሩን ይረግፉን፤ ጨናፍር እውን ጥራይ ይተርፉ።",
                "ኣብ ተኸታታሊ እዋናት ዝድገም ምርጋፍ ቅጠል ንገረብ የድክምን ናይ መጻኢ ዕምባባ ይንክይን።",
            ],
            "causes": "ብንፋስ ብዝመሓላለፍ ፈንገስ Hemileia vastatrix ዝስዕብ ኮይኑ፣ ብንፋስ፣ ብዝናብ ብናብ ምርጭት፣ ብባልዕን ብዝረኸሱ መሳርሒታት/ኣእዳውን ኣብ መንጎ ግራታት ይላባዕ።",
            "favorable_conditions": [
                "ውዑይ ደረጃ ሙቐት (21-25°ሴ) ምስ ልዑል ልሙዕነት ወይ ልሙዕነት ቅጠል (ዝናብ ወይ ከቢድ ኣኻውሒ) ተደሚሩ።",
                "ትሑትን ማእከላይን ብራኸ ዘለዎም ግራታት ካብ ልዑል ብራኸ ብዝበለጸ ተጋላጺ እዮም።",
                "ንቅጠል ንነዊሕ ግዜ ልሙዕ ዘጽንሕ ጥቅጥቅ ዝበለ ተኸላ/ከቢድ ጽላል።",
                "ብፍላይ ድሕሪ ከቢድ ናይ ቀደም ምርት ድኹማት ወይ ብንጥረ-ነገር ጭንቀት ዘለዎም ኣትክልቲ።",
            ],
            "immediate_actions": [
                "ኣብ ጥቓ ዘለዉ ቅጠላት ካብ ትሕቲ ብምርኣይ ናይቲ ባህርያዊ ብርቱካናዊ ሓመድ ምህላው ኣረጋግጽ።",
                "ናይ ስፖር ክምችት ንምንካይ ኣዝዮም ዝተጠቕዑ ቅጠላትን ጨናፍርን ቆርጽ — ካብ ግራት ርሒቕካ ኣወግድ፣ ሓድሽ ነገር ኮምፖስት ኣይትግበር።",
                "ብዘይ ዝኾነ ደንጓይ ናይ ጸረ-ፈንገስ ፕሮግራም ጀምር — ኩነታት ምቹእ ምስ ኮነ ዝገት ብቕልጡፍ ይላባዕ፤ ኣብ መጀመርታ ምቁጽጻር ካብ ደንጒኻ ምቁጽጻር ኣዝዩ ይሓይሽ።",
            ],
            "organic_treatment": [
                "ቅድሚ/ኣብ መጀመርታ ክራማት ብመዳብ ዝተሰርሐ ናይ ምክልኻል ርጭት (ኮፐር ሃይድሮክሳይድ፣ ኮፐር ኦክሲክሎራይድ ወይ ቦርዶ ድብልቕ) ተጠቐም።",
                "ብተመጣጣኒ ኦርጋኒክ ድዂዒ (ኮምፖስት፣ ድዂዒ) ከምኡውን እኹል ፖታሽየም ናይ ተኽሊ ጥንካረ ኣደልድል — ጠንካራ ኣትክልቲ ንዝገት ብዝበለጸ ይጻወሩ።",
                "ልሙዕነት ንምንካይ ጽላል ግረዝን ርሕቀት ወስኽን።",
            ],
            "chemical_treatment": [
                "ዝገት ናብ ልዑል ደረጃ ቅድሚ ምብጽሑ ከም መከላኸሊ/ኣቐዲሙ-መፍወሲ ብከባቢ ዝምከር መጠን ዝተመዝገበ ትሪያዞል ወይ ስትሮቢሉሪን ዓይነት ስርዓታዊ ጸረ-ፈንገስ ንፉግ።",
                "ናይ ፈንገስ ምክልኻል ንምድንጓይ ኣብ ውሽጢ እዋን ዝተፈላለዩ ናይ ጸረ-ፈንገስ ኬሚካል ክፍልታት ብተራ ተጠቐም።",
                "ንብዙሕ ግዜ ንዝተጠቕዑ ግራታት ንመጻኢ ተኽላ ናይ ዝገት-ተጻዋርነት ዝርያታት ቡን ኣብ ግምት ኣእቱ።",
                "ንመጠን፣ ንናይ ርጭት ፍልልይን ቅድሚ ምህርቲ ዝጽበ ግዜን ኣብ መለለዪ ዘሎ መምርሒ ብትኽክል ተኸተል።",
            ],
            "prevention": [
                "ብእተኻእለ መጠን ብሃገራዊ ትካላት ምርምር ቡን ብዝምከሩ ናይ ዝገት-ተጻዋርነት ዝርያታት ትከል/ደጊምካ ትከል።",
                "ማእከላይ ጽላል ሓሉ (ሙሉእ ጸሓያዊ ጭንቀትን ልዕሊ ዓቐን ጥቅጥቅ ዝበለ ልሙዕ ጽላልን ኣወግድ)።",
                "ትርፊ ናይትሮጅን ኣወግድ፣ እኹል ፖታሽየም ኣረጋግጽ — ብጽቡቕ ዝተመገቡ ኣትክልቲ ንዝገት ብዝበለጸ ይጻወሩ።",
                "ብፍላይ ኣብ መጀመርታ ክራማት ብስሩዕ ተዓዘብ፤ ኣብ መጀመርታ ደረጃ ዘለዉ ልበዳታት ብዘይ ምድንጓይ ስጉምቲ ውሰድ።",
                "ኣትክልቲ ድሕሪ ከቢድ ምርቲ ልዕሊ ዓቐን ጽዕነት ከይረኽቡ ብምርጫዊ መቖርቆር ናይ ምርቲ ጽዕነት ኣመሓድር።",
            ],
            "economic_note": "ዝገት ቅጠል ቡን ኣብ ብዙሓት ሃገራት ሃገራዊ ደረጃ ቅልውላው ምርቲ ኣስዒቡ እዩ፤ ማእከላይ ወረርሽን እውን ኣብ ሓደ እዋን ምርቲ ብ20-50% ክንክይ ይኽእል፣ ናብ ዝቕጽል ዓመት እውን ጽልዋ ከስፍሕ ይኽእል — እዚ ሕማም ህጹጽን ንጡፍን ኣተሓሕዛ ይደሊ፣ ጸኒሕካ ናይ ምርኣይ ኣገባብ ኣየድልዮን።",
            "when_to_seek_help": "ዝገት ልዕሊ 10-15% ኣብ ዘለዉ ቅጠላት እንተ ተረኺቡ ወይ ብቕልጡፍ ኣብ ግራት እናተላብዐ እንተኾይኑ፣ ውሁድን ብከባቢ ዝተፈቐደን ናይ ጸረ-ፈንገስ ፕሮግራም ንምርካብ ብህጹጽ ናይ ሕርሻ ኤክስቴንሽን ኣገልግሎት ወይ ማሕበር ቡን ኣዘራርብ።",
        },
        "om": {
            "name": "Roostii Baala Bunaa (Coffee Leaf Rust)",
            "local_name": "Roostii",
            "severity": "critical",
            "overview": "Roostiin Baala Bunaa fangaasii Hemileia vastatrix irraa kan dhufu yoo ta'u, addunyaa irratti dhukkuba buna diinagdeedhaan balaa guddaa geessisu isa tokkicha. Baalota kallattiin waan miidhuuf, waggoota 1-2f oomisha balleessuu danda'a; yoo hin bulchamne yeroo dheeraa keessatti mukoota ni ajjeesa.",
            "symptoms": [
                "Tuqaawwan xixiqqoo, bifa keelloo laafaa, akka zayitaa fakkaatan olitti baalaa irratti.",
                "Bakka wal-simuu jalaa BAALAA irratti daakuu bifa burtukaanaa-keelloo ifa ta'e — kun mallattoo addaan baasuu ijoo dha.",
                "Yeroo dhukkubni itti fufu, tuqaaleen ni guddatu, walittis ni makamu.",
                "Baalonni cimoo miidhaman gara keelloo/magaalaatti jijjiiramanii harca'u; damewwan qullaa hafu.",
                "Harca'uun irra deddeebi'ee waggoota aanan keessatti mudatu, muka guutuu dadhabsiisee daraaraa gara fuulduraa hir'isa.",
            ],
            "causes": "Fangaasii qilleensaan babal'atu Hemileia vastatrix irraa kan dhufu, qilleensa, bishaan bokkaa cabsuu, ilbiisota, fi meeshaa/harka xurawaa gidduu qonnaan bultootaatti babal'ata.",
            "favorable_conditions": [
                "Ho'a giddu galeessaa (21-25°C) jiidha ol'aanaa ykn jiidha baalaa (bokkaa ykn fixeensa) waliin ta'e.",
                "Naannoo gad aanaa fi giddu galeessaa ol'aantummaa caalaa balaaf saaxilamu.",
                "Dhaabuu walitti dhiyaate/gaaddisa hamaa kan baala jiidhaa dheeraaf qabu.",
                "Mukkeen dadhaboo ykn soorata hanqatan, keessumaa oomisha guddaa booda ('overbearing').",
            ],
            "immediate_actions": [
                "Baalota naannoo jiran jalaa ilaaluun daakuu burtukaanaa mirkaneessuun sirrii ta'uu isaa mirkaneessi.",
                "Sporii hir'isuuf baalotaa fi damewwan cimoo miidhaman mur'ii — dirree irraa fageessuun gata, waan haaraa koompostii hin taasisin.",
                "Utuu hin turin sagantaa fangaasii-ajjeesaa jalqabi — roostiin yeroo haalli mijate dafee babal'ata; wal'aansi yeroo jalqabaa kan booda irra caalaa milkaawaadha.",
            ],
            "organic_treatment": [
                "Utuu waqtiin bokkaa hin jalqabin/jalqaba isaatti ittisa koopparii irratti hundaa'e facaasi (fkn koopar haayidiroksaayidii, koopar oksikooloraayidii, ykn makaa Boordoo).",
                "Jabina biqiltuu soorata ogaanikii madaalawaa (koompostii, kosii) fi phootaashiyeemii gahaa ta'een jabeessi — mukkeen jabaan roostii caalaatti dandamatu.",
                "Jiidha hir'isuuf gaaddisa qeenxi, fageenyas dabali.",
            ],
            "chemical_treatment": [
                "Utuu roostiin hammaachaa hin jirre dura akka ittisaatti/wal'aansa jalqabaa fangaasii-ajjeesaa gosa traayaazoolii ykn istiroobiluriinii safara naannootti hayyamameen facaasi.",
                "Akka fangaasiin dandamachuu hin ijaarrannee waqtii keessa gosoota kimikaalaa fangaasii-ajjeesaa garaagaraa jijjiiraa fayyadami.",
                "Dirreewwan yeroo hunda miidhaman keessatti dhaloota buna roostii dandamatu bocuu fuulduraaf yaadi — kun furmaata yeroo dheeraa cimaa dha.",
                "Safara, hafuura facaasii fi yeroo eegumsa oomisha lebeela irratti barreeffame sirriitti hordofi.",
            ],
            "prevention": [
                "Yoo danda'ame dhaloota roostii dandamatu kan dhaabbileen qorannoo buna biyyaalessaa gorsan facaasi/haaromsi.",
                "Gaaddisa giddu galeessaa eegi (dhiibbaa aduu guutuu fi gaaddisa jiidhaa baay'ee walitti dhiyaate lachuu irraa fagaadhu).",
                "Naayitrojiinii dabalataa irraa fagaadhu; phootaashiyeemii gahaa mirkaneessi — mukkeen sirriitti nyaataman roostii caalaatti dandamatu.",
                "Yeroo hunda, keessumaa jalqaba waqtii bokkaa, sakatta'i; tartiiba jalqabaa irratti utuu hin turin tarkaanfii fudhadhu.",
                "Mukkeen booda oomisha guddaa akka hin dhiphanne, muraa filatamaadhaan baay'ina oomishaa bulchi.",
            ],
            "economic_note": "Roostiin Baala Bunaa biyyoota hedduu keessatti rakkoo oomishaa sadarkaa biyyaalessaa fideera; babal'inni giddu galeessaa iyyuu waqtii tokko keessatti oomisha %20-50 hir'isuu danda'a, waggaa itti aanuttis dhiibbaa dabaluu danda'a — dhukkubni kun bulchiinsa ariifachiisaa fi kan of eeggannoo qabu barbaada; malli eeggachiisaa hin barbaachisu.",
            "when_to_seek_help": "Roostiin %10-15 ol baalota irratti yoo argame, ykn dirree keessa dafee babal'achaa yoo jiraate, sagantaa fangaasii-ajjeesaa naannootti hayyamame walta'iinsaan argachuuf tajaajila ergaa qonnaa naannoo ykn waldaa buna naannoo keetii utuu hin turin dhaqqabi.",
        },
    },

    # ==================================================================
    # CERCOSPORA LEAF SPOT
    # ==================================================================
    "cercospora": {
        "en": {
            "name": "Cercospora Leaf Spot",
            "local_name": "Brown Eye Spot / 'Cercospora coffeicola'",
            "severity": "moderate",
            "overview": "Cercospora Leaf Spot, caused by the fungus Cercospora coffeicola, produces distinctive circular 'brown eye' spots on leaves and can also affect berries ('Berry Blotch'). It is generally less destructive than Rust or Phoma but is often a strong indicator of underlying nutrient deficiency or excessive sun stress that needs correcting.",
            "symptoms": [
                "Circular, brown spots with a distinctive tan/gray center and a dark brown to reddish-yellow (chlorotic) ring — the 'eye-spot' or 'target' appearance.",
                "Spots typically 3-10mm in diameter, scattered across the leaf, sometimes merging in severe cases.",
                "Heavily spotted leaves yellow and drop prematurely, especially lower/older leaves first.",
                "On berries: sunken brown-black lesions ('berry blotch') that can cause premature ripening and bean quality loss.",
            ],
            "causes": "Caused by the fungus Cercospora coffeicola. Outbreaks are strongly associated with plant stress — particularly nitrogen or potassium deficiency, excessive direct sunlight (insufficient shade), and moisture stress.",
            "favorable_conditions": [
                "Nutrient-deficient soils, especially low nitrogen or potassium.",
                "Excessive sun exposure from insufficient or removed shade cover.",
                "Warm, humid conditions with extended leaf wetness.",
                "Trees weakened by drought stress, overbearing, or poor overall management.",
            ],
            "immediate_actions": [
                "Assess overall plant nutrition and shade levels first — Cercospora often signals an underlying management issue, not just a standalone pathogen problem.",
                "Remove and destroy heavily spotted fallen leaves around the base of affected trees to reduce spore carryover.",
                "If berries show lesions, prioritize harvesting and sorting affected cherries separately to protect overall lot quality.",
            ],
            "organic_treatment": [
                "Correct nutrition first: apply balanced compost/manure with attention to nitrogen and potassium; consider a soil or leaf-tissue test if available.",
                "Adjust shade to the recommended 40-60% cover to reduce direct sun stress on foliage.",
                "Apply copper-based fungicide as a protective spray in persistent or severe cases.",
            ],
            "chemical_treatment": [
                "In moderate-to-severe outbreaks, apply a registered copper-based or broad-spectrum fungicide following local label guidance, prioritizing the youngest, most productive branches.",
                "Combine fungicide application with a corrective fertilization program — treating the fungus alone without fixing the underlying nutrient stress often leads to recurrence.",
            ],
            "prevention": [
                "Maintain balanced fertilization, particularly adequate nitrogen and potassium, based on soil testing where possible.",
                "Keep shade at the recommended 40-60% cover — neither excessive sun nor overly dense shade.",
                "Avoid drought stress where possible with mulching to conserve soil moisture.",
                "Manage crop load through pruning to avoid overbearing, which depletes the tree's nutrient reserves and increases susceptibility.",
                "Scout regularly and treat the nutritional cause early rather than only reacting to visible spots.",
            ],
            "economic_note": "While rarely fatal on its own, chronic Cercospora infection reduces photosynthetic leaf area and can directly downgrade cherry and bean quality through berry blotch — correcting the underlying nutrition issue often resolves the disease and improves yield simultaneously.",
            "when_to_seek_help": "If spotting is widespread despite corrected shade and fertilization, or if berry blotch is significantly affecting your harvest, consult a local agronomist to rule out compounding factors such as soil pH imbalance or additional pest pressure.",
        },
        "am": {
            "name": "የሰርኮስፖራ ቅጠል ነጠብጣብ (Cercospora Leaf Spot)",
            "local_name": "ቡናማ-ዓይን ነጠብጣብ",
            "severity": "moderate",
            "overview": "የሰርኮስፖራ ቅጠል ነጠብጣብ በCercospora coffeicola ፈንገስ የሚከሰት ሲሆን በቅጠሎች ላይ ልዩ የሆነ ክብ 'ቡናማ-ዓይን' ነጠብጣብ ያመጣል፤ በፍሬዎችም ላይ ሊከሰት ይችላል። በአጠቃላይ ከዝገት ወይም ከፎማ ያነሰ ጎጂ ቢሆንም ብዙውን ጊዜ መስተካከል ያለበት የንጥረ ነገር እጥረት ወይም ከልክ ያለፈ የፀሐይ ጫና ምልክት ነው።",
            "symptoms": [
                "ክብ ቡናማ ነጠብጣቦች ልዩ የሆነ ግራጫ/ቡናማ ማዕከልና ጥቁር ቡናማ ወደ ቀይ-ቢጫ (ክሎሮቲክ) ቀለበት ያላቸው — 'የዓይን-ነጠብጣብ' መልክ።",
                "ነጠብጣቦቹ በአብዛኛው 3-10ሚሜ ዲያሜትር ያላቸው ሲሆኑ በቅጠሉ ላይ ተበታትነው ይገኛሉ፤ በከባድ ሁኔታ ላይ ሊዋሃዱ ይችላሉ።",
                "በጣም የነጠበጠቡ ቅጠሎች ወደ ቢጫ ይቀየራሉና ከጊዜው በፊት ይረግፋሉ፤ በተለይ የታችኛዎቹ/አሮጌዎቹ ቅጠሎች መጀመሪያ።",
                "በፍሬዎች ላይ፦ ጎድጓዳ ቡናማ-ጥቁር ነጠብጣቦች ('berry blotch') ያለ ጊዜው መብሰልንና የፍሬ ጥራት መቀነስን ያስከትላሉ።",
            ],
            "causes": "በCercospora coffeicola ፈንገስ ይከሰታል። ወረርሽኙ በተለይ ከናይትሮጅን ወይም ፖታሺየም እጥረት፣ ከልክ ያለፈ ቀጥተኛ የፀሐይ ብርሃን (በቂ ጥላ አለመኖር) እና ከውሃ እጥረት ጋር በጥብቅ የተያያዘ ነው።",
            "favorable_conditions": [
                "በተለይ ዝቅተኛ ናይትሮጅን ወይም ፖታሺየም ያለው የንጥረ ነገር እጥረት ያለበት አፈር።",
                "በቂ ያልሆነ ወይም የተወገደ ጥላ ሽፋን ያለበት ከልክ ያለፈ የፀሐይ መጋለጥ።",
                "ሙቅና እርጥብ ሁኔታ ከረዘመ የቅጠል እርጥበት ጋር።",
                "በድርቅ ጫና፣ ከልክ ያለፈ ምርት ወይም ደካማ አጠቃላይ አያያዝ የተዳከሙ ዛፎች።",
            ],
            "immediate_actions": [
                "በመጀመሪያ አጠቃላይ የተክሉን ንጥረ ነገርና የጥላ መጠን ይገምግሙ — ሰርኮስፖራ ብዙውን ጊዜ የአያያዝ ችግርን እንጂ ብቻውን የፈንገስ ችግርን አያመለክትም።",
                "የስፖር ክምችትን ለመቀነስ በጣም የነጠበጠቡ የረገፉ ቅጠሎችን ከተጠቁ ዛፎች ስር ያስወግዱና ያጥፉ።",
                "ፍሬዎች ነጠብጣብ ካሳዩ፣ የጠቅላላውን ምርት ጥራት ለመጠበቅ የተጠቁ ፍሬዎችን ለብቻ በቅድሚያ ይሰብስቡና ይለዩ።",
            ],
            "organic_treatment": [
                "መጀመሪያ ንጥረ ነገርን ያስተካክሉ፦ ለናይትሮጅንና ፖታሺየም ትኩረት በመስጠት ሚዛናዊ ኮምፖስት/ፍግ ይጠቀሙ፤ ከተቻለ የአፈር ወይም የቅጠል ምርመራ ያድርጉ።",
                "በቅጠሉ ላይ ያለውን ቀጥተኛ የፀሐይ ጫና ለመቀነስ ጥላን ወደ 40-60% ሽፋን ያስተካክሉ።",
                "በተደጋጋሚ ወይም በከባድ ሁኔታ ላይ የመዳብ ተኮር ፀረ-ፈንገስ በመከላከያ መልክ ይርጩ።",
            ],
            "chemical_treatment": [
                "በመካከለኛ-እስከ-ከባድ ወረርሽኝ ላይ፣ በአካባቢው መለያ መመሪያ መሰረት የተመዘገበ የመዳብ ተኮር ወይም ሰፊ-ስፔክትረም ፀረ-ፈንገስ ይርጩ፤ ለወጣትና ምርታማ ቅርንጫፎች ቅድሚያ ይስጡ።",
                "የፀረ-ፈንገስ ርጭትን ከማስተካከያ ማዳበሪያ ፕሮግራም ጋር ያጣምሩ — ፈንገሱን ብቻ ማከምና የስር ንጥረ ነገር ጭንቀትን ያለማስተካከል ብዙውን ጊዜ ወደ ድጋሚ ማገርሸት ያመራል።",
            ],
            "prevention": [
                "ሲቻል በአፈር ምርመራ ላይ ተመስርተው ለናይትሮጅንና ፖታሺየም በቂ ትኩረት የሰጠ ሚዛናዊ ማዳበሪያ ይጠቀሙ።",
                "ጥላን በ40-60% ሽፋን ደረጃ ይጠብቁ — ከልክ ያለፈ ፀሐይም ሆነ ከልክ ያለፈ ጥቅጥቅ ያለ ጥላ አያስፈልግም።",
                "በተቻለ መጠን አፈር እርጥበት ለመጠበቅ ሙልች በመጠቀም የድርቅ ጫናን ያስወግዱ።",
                "የዛፉን ንጥረ ነገር ክምችት የሚያሟጥጥና ተጋላጭነትን የሚጨምር ከልክ ያለፈ ምርትን ለማስወገድ በመከርከም የምርት ጫናን ያስተዳድሩ።",
                "በመደበኛነት ይቃኙና ለሚታዩት ነጠብጣቦች ብቻ ከመምላስ ይልቅ የስር ንጥረ-ነገር መንስኤውን ቀድመው ያክሙ።",
            ],
            "economic_note": "ብቻውን እምብዛም ገዳይ ባይሆንም፣ ተከታታይ የሰርኮስፖራ ኢንፌክሽን የፎቶሲንተሲስ ቅጠል ስፋትን ይቀንሳልና በፍሬ ነጠብጣብ አማካኝነት የፍሬ ጥራትን በቀጥታ ሊያሳንስ ይችላል — የስር ንጥረ-ነገር ችግርን ማስተካከል ብዙውን ጊዜ በሽታውን ይፈታልና ምርትንም በተመሳሳይ ያሻሽላል።",
            "when_to_seek_help": "ጥላና ማዳበሪያ ካስተካከሉ በኋላም ነጠብጣቡ በሰፊው ከቀጠለ ወይም የፍሬ ነጠብጣብ ምርትዎን በእጅጉ እየጎዳ ከሆነ፣ እንደ አፈር ፒኤች አለመመጣጠን ወይም ተጨማሪ የተባይ ጫና ያሉ ተደራራቢ ምክንያቶችን ለማስወገድ የአካባቢ ግብርና ባለሙያ ያማክሩ።",
        },
        "ti": {
            "name": "ሰርኮስፖራ ነጠብጣብ ቅጠል (Cercospora Leaf Spot)",
            "local_name": "ቡናዊ-ዓይኒ ነጠብጣብ",
            "severity": "moderate",
            "overview": "ሰርኮስፖራ ነጠብጣብ ቅጠል ብፈንገስ Cercospora coffeicola ዝስዕብ ኮይኑ፣ ኣብ ቅጠላት ፍሉይ ዙርያዊ 'ቡናዊ-ዓይኒ' ነጠብጣብ የምጽእ፤ ኣብ ፍረታት እውን ክርከብ ይኽእል። ብሓፈሻ ካብ ዝገት ወይ ፎማ ዝወሓደ ጉድኣት እኳ እንተኾነ፣ መብዛሕትኡ ግዜ ክእረም ዘለዎ ናይ ንጥረ-ነገር ሕጽረት ወይ ልዕሊ ዓቐን ጸሓያዊ ጭንቀት ምልክት እዩ።",
            "symptoms": [
                "ዙርያዊ ቡናዊ ነጠብጣብ ብፍሉይ ሓመዅሽታዊ/ግራጫ ማእከልን ጸሊም ቡናዊ ናብ ቀይሕ-ብጫ (ክሎሮቲክ) ቀለበትን ዘለዎ — ናይ 'ዓይኒ-ነጠብጣብ' ትርኢት።",
                "ነጠብጣባት መብዛሕትኡ ግዜ 3-10ሚሜ ዲያሜተር ዘለዎም ኮይኖም ኣብ ቅጠል ተበታቲኖም ይርከቡ፤ ኣብ ከቢድ ኩነታት ክላገቡ ይኽእሉ።",
                "ኣዝዮም ዝነጠብጠቡ ቅጠላት ናብ ብጫ ይቕየሩን ቅድሚ ግዜኦም ይረግፉን፤ ብፍላይ ታሕተዎት/ኣረጋውያን ቅጠላት ቅድም።",
                "ኣብ ፍረታት፦ ጎዳጉድ ቡናዊ-ጸሊም ነጠብጣብ ('berry blotch') ቅድሚ ግዜኡ ምብሳልን ናይ ፍረ ጽሬት ምንካይን ከኸትል ይኽእል።",
            ],
            "causes": "ብፈንገስ Cercospora coffeicola ይስዕብ። ወረርሽን ብፍላይ ምስ ናይትሮጅን ወይ ፖታሽየም ሕጽረት፣ ልዕሊ ዓቐን ቀጥታዊ ጸሓያዊ ብርሃን (ብዘይ እኹል ጽላል) ከምኡውን ናይ ማይ ጭንቀት ተኣሳሲሩ ይርከብ።",
            "favorable_conditions": [
                "ብፍላይ ትሑት ናይትሮጅን ወይ ፖታሽየም ዘለዎ ናይ ንጥረ-ነገር ሕጽረት ዘለዎ ሓመድ።",
                "ብዘይ እኹል ወይ ዝተወገደ ጽላል ሽፋን ዝፍጠር ልዕሊ ዓቐን ጸሓያዊ ምቅላዕ።",
                "ውዑይን ልሙዕን ኩነታት ምስ ዝነውሐ ልሙዕነት ቅጠል።",
                "ብድርቂ ጭንቀት፣ ልዕሊ ዓቐን ምርቲ ወይ ድኹም ሓፈሻዊ ኣተሓሕዛ ዝደኸሙ ኣትክልቲ።",
            ],
            "immediate_actions": [
                "መጀመርታ ሓፈሻዊ ናይ ተኽሊ ንጥረ-ነገርን ደረጃ ጽላልን ገምግም — ሰርኮስፖራ መብዛሕትኡ ግዜ ናይ ኣተሓሕዛ ጸገም እምበር ንበይኑ ናይ ፈንገስ ጸገም ኣየመልክትን።",
                "ናይ ስፖር ክምችት ንምንካይ ኣዝዮም ዝነጠብጠቡ ዝረገፉ ቅጠላት ካብ ስረ ዝተጠቕዑ ኣትክልቲ ኣወግድን ኣጥፍእን።",
                "ፍረታት ነጠብጣብ እንተ ኣርእዮም፣ ንሓፈሻዊ ጽሬት ምርቲ ንምዕቃብ ንዝተጠቕዑ ፍረታት ብቕድሚያ ንበይኖም ኣኪብካ ፈልዮም።",
            ],
            "organic_treatment": [
                "መጀመርታ ንጥረ-ነገር ኣረም፦ ንናይትሮጅንን ፖታሽየምን ኣቓልቦ ብምሃብ ተመጣጣኒ ኮምፖስት/ድዂዒ ተጠቐም፤ እንተተኻኢሉ ናይ ሓመድ ወይ ቅጠል መርመራ ግበር።",
                "ኣብ ቅጠል ዘሎ ቀጥታዊ ጸሓያዊ ጭንቀት ንምንካይ ጽላል ናብ 40-60% ሽፋን ኣተዓራሪ።",
                "ኣብ ተደጋጋሚ ወይ ከቢድ ኩነታት ብመዳብ ዝተሰርሐ ጸረ-ፈንገስ ከም መከላኸሊ ንፉግ።",
            ],
            "chemical_treatment": [
                "ኣብ ማእከላይ-ክሳብ-ከቢድ ወረርሽን፣ ብመሰረት ናይ ከባቢ መለለዪ መምርሒ ዝተመዝገበ ብመዳብ ዝተሰርሐ ወይ ሰፊሕ-ስፔክትረም ጸረ-ፈንገስ ንፉግ፤ ንመንእሰያትን ፍርያምን ጨናፍር ቅድሚያ ሃብ።",
                "ናይ ጸረ-ፈንገስ ርጭት ምስ ናይ ምእራም ድዂዒ ፕሮግራም ኣተኣሳስር — ንፈንገስ ጥራይ ምሕካምን ናይ ስረ ንጥረ-ነገር ጭንቀት ዘይምእራምን መብዛሕትኡ ግዜ ናብ ድግማ ተመላሲ ይመርሕ።",
            ],
            "prevention": [
                "እንተተኻኢሉ ብመርመራ ሓመድ ተሰሪቱ ንናይትሮጅንን ፖታሽየምን እኹል ኣቓልቦ ዝሃበ ተመጣጣኒ ድዂዒ ተጠቐም።",
                "ጽላል ኣብ 40-60% ሽፋን ደረጃ ሓሉ — ልዕሊ ዓቐን ጸሓይን ልዕሊ ዓቐን ጥቅጥቅ ዝበለ ጽላልን ኣየድልን።",
                "ብእተኻእለ መጠን ልሙዕነት ሓመድ ንምዕቃብ ሙልች ብምጥቃም ናይ ድርቂ ጭንቀት ኣወግድ።",
                "ንንጥረ-ነገር ክምችት ገረብ ዘሟጥጥን ተጋላጽነት ዝውስኽን ልዕሊ ዓቐን ምርቲ ንምውጋድ ብመቖርቆር ናይ ምርቲ ጽዕነት ኣመሓድር።",
                "ብስሩዕ ተዓዘብን ንዝረኣዩ ነጠብጣብ ጥራይ ካብ ምምላስ ይልቅ ንናይ ስረ ንጥረ-ነገር ምኽንያት ኣቐዲምካ ኣክም።",
            ],
            "economic_note": "ንበይኑ ብሒደት ገዳይ እኳ እንተዘይኮነ፣ ተኸታታሊ ናይ ሰርኮስፖራ ልበዳ ናይ ፎቶሲንተሲስ ቦታ ቅጠል ይንክይን ብፍረ-ነጠብጣብ ኣቢሉ ናይ ፍረ ጽሬት ብቐጥታ ከንክይ ይኽእልን — ናይ ስረ ንጥረ-ነገር ጸገም ምእራም መብዛሕትኡ ግዜ ንሕማም ይፈትሕን ምርቲ እውን የመሓይሽን።",
            "when_to_seek_help": "ጽላልን ድዂዕን ድሕሪ ምእራምካ እውን ነጠብጣብ ብሰፊሑ እንተቐጺሉ ወይ ናይ ፍረ-ነጠብጣብ ምርትኻ ብትርኢት እናጎድአ እንተኾይኑ፣ ከም ናይ ሓመድ pH ዘይምዕሩይነት ወይ ተወሳኺ ጸቕጢ ባልዕ ዝኣመሰሉ ተደራረብቲ ምኽንያታት ንምውጋድ ናይ ከባቢ ክኢላ ሕርሻ ኣማኽር።",
        },
        "om": {
            "name": "Tuqaa Baala Sarkoosporaa (Cercospora Leaf Spot)",
            "local_name": "Tuqaa Ija-Magaalaa",
            "severity": "moderate",
            "overview": "Tuqaan Baala Sarkoosporaa, kan fangaasii Cercospora coffeicola irraa dhufu, baala irratti tuqaa 'ija-magaalaa' addaa geengoo ta'e uuma; firii irrattis dhiibbaa geessisuu danda'a. Waliigala Roostii ykn Fommaa caalaa miidhaa xiqqaa qaba, garuu yeroo baay'ee mallattoo hanqina soorataa ykn dhiibbaa aduu dabalataa sirreeffamuu qabu agarsiisa.",
            "symptoms": [
                "Tuqaawwan geengoo, bifa magaalaa kan giddugala bifa suphee/daalachaa fi naannoo hidhata bifa magaalaa-diimaa hanga keelloo (kiloorootikii) qabu — mul'ina 'tuqaa-ijaa'.",
                "Tuqaaleen bal'inaan 3-10mm ta'anii baala irratti facaʼanii argamu; haala hamaa keessatti walitti makamuu danda'u.",
                "Baalonni baay'ee tuqaan qaban ni keellawu, yeroo hin gahin ni harca'u — keessumaa baalonni gadii/dulloomoon dursanii.",
                "Firii irratti: madaa gurraacha-magaalaa yaachaa ('berry blotch') kan bilchina yeroo hin geenye fi hir'ina qulqullina oomishaa fidu.",
            ],
            "causes": "Fangaasii Cercospora coffeicola irraa dhufa. Babal'inni isaa dhiibbaa biqiltuu waliin walqabata — keessumaa hanqina Naayitrojiinii ykn Phootaashiyeemii, ifa aduu kallattii dabalataa (gaaddisa gahaa dhabuu), fi dhiibbaa jiidhaa.",
            "favorable_conditions": [
                "Biyyee hanqina soorataa qabu, keessumaa Naayitrojiinii ykn Phootaashiyeemii gadi aanaa.",
                "Ifa aduu dabalataa gaaddisa gahaa dhabuu ykn balleeffamuu irraa dhufu.",
                "Haala ho'aa, jiidhaa fi jiidha baalaa dheeraa.",
                "Mukkeen hoongee, oomisha dabalataa, ykn bulchiinsa gadaanaa irraa dadhaboo ta'an.",
            ],
            "immediate_actions": [
                "Jalqaba soorata biqiltuu waliigalaa fi sadarkaa gaaddisaa madaali — Sarkoosporaan yeroo baay'ee rakkoo bulchiinsaa argisiisa, rakkoo fangaasii qofaa osoo hin taane.",
                "Sporii hir'isuuf baalota tuqaa baay'ee qaban kan harca'an naannoo hidda mukaa miidhamee irraa fageessuun balleessi.",
                "Yoo firiin madaa mul'ise, qulqullina oomisha waliigalaa eeguuf firii miidhaman filadhee cite addaan baasi.",
            ],
            "organic_treatment": [
                "Jalqaba soorata sirreessi: koompostii/kosii madaalawaa Naayitrojiinii fi Phootaashiyeemiif xiyyeeffannoo kennuun fayyadami; yoo danda'ame qorannoo biyyee ykn baalaa gochuu yaadi.",
                "Dhiibbaa aduu kallattii baala irratti hir'isuuf gaaddisa haguuggii %40-60 gara sirriitti sirreessi.",
                "Yeroo dhukkubni itti fufe ykn hamaate fangaasii-ajjeesaa koopparii irratti hundaa'e akka ittisaatti facaasi.",
            ],
            "chemical_treatment": [
                "Babal'ina giddu galeessaa hanga hamaa keessatti, fangaasii-ajjeesaa sibiila (koopparii) irratti hundaa'e ykn bal'aa ta'e labeelii naannoo hordofuun facaasi; damee dargaggeessaa fi oomishaawaa caalaa dursi.",
                "Facaasii fangaasii-ajjeesaa sagantaa xanxala sirreessuu waliin walitti fidi — fangaasii qofa wal'aanuu fi dhiibbaa soorata jalaa hin sirreessine yeroo baay'ee deebi'ee mudata.",
            ],
            "prevention": [
                "Yoo danda'ame qorannoo biyyee irratti hundaa'e, xanxala madaalawaa keessumaa Naayitrojiinii fi Phootaashiyeemii gahaa qabu fayyadami.",
                "Gaaddisa haguuggii %40-60 eegi — aduu dabalataa yookaan gaaddisa dabalataa lachuu hin barbaachisu.",
                "Yoo danda'ame jiidha biyyee eeguuf caccabaa (mulchii) fayyadamuun dhiibbaa hoongee hir'isi.",
                "Kuufama soorataa mukaa fixuu fi saaxilamummaa dabaluu irraa fagaachuuf, muraa filatamaadhaan baay'ina oomishaa bulchi.",
                "Yeroo hunda sakatta'i; sababa soorataa jalaa jiru dursii sirreessi, tuqaa mul'atu qofa deebii hin kennin.",
            ],
            "economic_note": "Kophaa isaatti dhukkuba nama ajjeesu ta'uun hin baay'atu, garuu dhukkubni Sarkoosporaa yeroo dheeraa itti fufu bal'ina baalaa hoosisaa ifaa hir'isee firii-madaa (berry blotch) tiin qulqullina firii fi bunaa kallattiin gadi buusuu danda'a — rakkoo soorataa jalaa jiru sirreessuun yeroo baay'ee dhukkuba furee oomishas ni fooyyessa.",
            "when_to_seek_help": "Gaaddisaa fi xanxala sirreessite booda iyyuu tuqaan yoo bal'inaan itti fufe, ykn madaan firii oomisha kee dhiibbaa guddaa yoo geessise, wantoota walxaxaa akka madaalii pH biyyee ykn dhiibbaa ilbiisaa dabalataa ilaaluuf ogeessa qonnaa naannoo mari'adhu.",
        },
    },
}


LANGUAGES = {
    "en": {"label": "English", "native": "English", "flag_emoji": "🇬🇧"},
    "am": {"label": "Amharic", "native": "አማርኛ", "flag_emoji": "🇪🇹"},
    "ti": {"label": "Tigrinya", "native": "ትግርኛ", "flag_emoji": "🇪🇹"},
    "om": {"label": "Afaan Oromoo", "native": "Afaan Oromoo", "flag_emoji": "🇪🇹"},
}

DEFAULT_LANGUAGE = "en"


def get_report(class_key: str, lang: str) -> dict:
    """
    Return the full advisory report for a given canonical class key
    and language code, falling back to English if the language or
    class is not recognised.
    """
    class_block = CONTENT.get(class_key)
    if class_block is None:
        class_block = CONTENT["not_coffee_leaf"]
    return class_block.get(lang) or class_block.get(DEFAULT_LANGUAGE)
