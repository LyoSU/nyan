import json
import logging
import time

import pytest

from nyan.rubrics import count_own_posts, RubricDetector

# The cases below are real posts, collected from the channels in channels.json.
# They are here because the first version of these patterns matched plain
# substrings and quietly swallowed news: "щирі співчуття" appears at the end of
# genuine reports of shelling, and "церемонія прощання" appears inside the
# president's evening address. Every entry in NEWS is a post an earlier pattern
# got wrong.


@pytest.fixture
def detector() -> RubricDetector:
    """The detector as production configures it, not a hand-built one."""
    with open("configs/annotator_config.json") as r:
        config = json.load(r)
    return RubricDetector(config["rubric_detector"])


# Routine: the daily ritual, a digest, a funeral notice.
ROUTINE = [
    "🕯 9:00 – загальнонаціональна хвилина мовчання Зупиніться й згадайте тих, "
    "чиї життя забрала війна. Військових і цивільних. Світла пам’ять!",
    "Щоранку о 9:00 – хвилина мовчання за всіма, хто загинув через російське "
    "вторгнення. Пам’ятаємо...",
    "Щоранку о 9:00 вшановуємо хвилиною мовчання пам’ять наших загиблих",
    "Щоранку о 9:00 хвилиною мовчання вшановуємо пам'ять загиблих у жорстокій "
    "війні росії проти України",
    "Сьогодні у Кривому Розі. Ранок. Хвилина мовчання",
    "Пам'ятати та вшанувати. Кожного та кожну. Івано-Франківськ. "
    "Загальнонаціональна хвилина мовчання",
    "Світла пам'ять загиблим. Вшануймо хвилиною мовчання.",
    "День пам'яті жертв Голодомору. Загальнонаціональна хвилина мовчання.",
    "Читайте наш щоденний дайджест найголовніших подій дня на випадок, якщо ви "
    "щось пропустили. Ось головне за 19 серпня.",
    "Дайджест новин: Сьогодні зранку окупанти вдарили ракетами по Миколаєву. "
    "В Одесі спокійно.",
    "Дайджест новини за ніч: Вночі українська ППО збила над Одещиною дві "
    "крилаті ракети.",
    "Прощання з Дмитром Коцюбайло відбудеться завтра в с.Бовшів о 12:00. "
    "Поховання відбудеться 10.03.2023 в Києві.",
    # The funeral notice from the screenshot that prompted all of this.
    "Сьогодні у Вінниці в останню путь проведуть добровольця Руслана Назаренка\n"
    "Захисник поліг у бою 14 вересня 2024-го поблизу присілка Руська Конопелька "
    "Курської області. Йому назавжди 37...\n"
    "О 9.15 — церемонія прощання на Янгеля, 4 («Реквієм Хол»).\n"
    "Об 11.00 — служба у Спасо-Преображенському соборі.\n"
    "О 12.00 — поховання на Алеї Слави Сабарівського кладовища.\n"
    "Редакція RIA/20 хвилин висловлює щирі співчуття рідним та близьким Героя.",
    # The air-raid stream: the siren, the all-clear, and the warnings that come
    # with them. A reader who needs these does not learn them from a digest an
    # hour later, and every channel posts them, so a cluster forms every time.
    "🔴 Повітряна тривога в Києві",
    "Повітряна тривога в Києві та Київській області. Прямуйте в укриття!",
    "Увага! Повітряна тривога в Харкові та області",
    "У Києві та області — повітряна тривога",
    # The cluster from the screenshot that prompted this.
    "У Києві оголосили повітряну тривогу, згодом відбій. 27 липня у Києві та "
    "низці областей оголосили повітряну тривогу через загрозу балістики. "
    "О 9:39 у столиці оголосили відбій.",
    "🟢 Відбій тривоги в Києві",
    "Відбій повітряної тривоги в Дніпрі та області",
    "Тривога в Полтавській області",
    "Загроза балістики! Негайно в укриття!",
    "Увага! Загроза застосування балістичного озброєння для Київщини",
    "Шахеди курсом на Полтавщину",
    "БпЛА курсом на Київ, прямуйте в укриття",
    "Негайно прямуйте в укриття — у місті працює ППО",
    "Зліт МіГ-31К — загроза застосування балістичного озброєння по всій Україні",
    # Posts that still reached the feed after the patterns above were written.
    # Every one of them names the place first and the event after a dash, which
    # is how a regional channel writes all of these, not only the siren.
    "🔴 Тульчинський район - повітряна тривога!",
    "🟢 Тульчинський район - відбій тривоги!",
    "‼️ Київ та низка областей — загроза балістики",
    "🟢 Відбій повітряної тривоги у Києві та низці областей",
    # The alert says where, and then why. The reason is part of the alert.
    "❗️Тривога в Києві та низці областей через загрозу БпЛА.",
    # Collected by running the detector over 262 live posts from thirty of the
    # channels in channels.json. Each block below is a format a whole channel
    # publishes many times a day, and every one of them was getting through.
    #
    # The passive voice: "оголошена" is not "оголосили", and the siren is
    # announced in the nominative, not the accusative the pattern asked for.
    "🚨УВАГА🚨\nОголошена повітряна тривога по всій Запорізькій області!\n"
    "Бережіть себе і терміново прослідуйте у безпечне місце!",
    "‼УВАГА! У Києві оголошена повітряна тривога!\n"
    "Просимо всіх терміново прослідувати в укриття цивільного захисту!\n"
    "Мапа укриттів - ‼ ATTENTION! Air raid sirens in Kyiv!\n"
    "Please proceed to the nearest shelter!",
    # The place on a line of its own, sometimes two of them, with the event
    # marker pressed straight up against the newline.
    "ЧЕРКАСЬКА ОБЛАСТЬ\n❗️❗️ПОВІТРЯНА ТРИВОГА (27.07/11:52)",
    "ЧЕРКАСЬКА ОБЛАСТЬ\n🟢ВІДБІЙ (27.07/12:07)",
    "Уманський район \nЗвенигородський район\n🟢ВІДБІЙ (28.07/00:29)",
    # And the same thing on one line, where the separator was an emoji the text
    # processor strips, leaving nothing but the space it sat in.
    "УВАГА!  КІРОВОГРАДСЬКА ОБЛАСТЬ  ВІДБІЙ ПОВІТРЯНОЇ ТРИВОГИ!",
    "УВАГА! КІРОВОГРАДСЬКА ОБЛАСТЬ ПОВІТРЯНА ТРИВОГА!",
    "УВАГА! ВІДБІЙ ПОВІТРЯНОЇ ТРИВОГИ в Новоукраїнському районі!\n"
    "УВАГА! ВІДБІЙ ПОВІТРЯНОЇ ТРИВОГИ в Голованівському районі!",
    # The air force channel tracks every drone, and only one of its phrasings
    # was "курсом на". A heading is a heading whichever way it is written.
    "Група реактивних БпЛА прямує на Ромни",
    "БпЛА на Запоріжжя з півдня",
    "Реактивний БпЛА у напрямку Барвінкового на Харківщині з південного заходу",
    "Реактивні БпЛА вздовж межі між Чернігівською та Полтавською областями "
    "південно-західним курсом",
    "Реактивні БпЛА на заході від Дніпра, на північно-східний напрямок.",
    # The minute of silence, introduced by a word rather than by the clock.
    "🕯️ О 9:00 — загальнонаціональна хвилина мовчання. Це щоденний ритуал "
    "подяки, пошани та пам’яті.\n\nУ цей момент країна завмирає, щоб вшанувати "
    "військових і цивільних, які загинули внаслідок збройної агресії.",
    "Щодня о 9:00 – загальнонаціональна хвилина мовчання…\n\n🕯Памʼятаємо кожного "
    "та кожну, чиє життя обірвали російські окупанти.",
    # Collected by asking six days of live posts which first lines a channel
    # repeats on four or more separate days. That question turns out to name the
    # rubrics on its own — the ritual, the siren, the digest — and it named these
    # thirty-eight, which no pattern here had caught.
    #
    # The daily remembrance, said without the words "хвилина мовчання". Nine in
    # the morning and a word for honouring the dead is the whole of it.
    "9.00 — Вшануймо пам’ять усіх, хто загинув внаслідок збройної агресії "
    "російської федерації проти України.",
    "Вшануймо пам'ять загиблих внаслідок збройної агресії російської федерації "
    "проти України.",
    "Щодня о 09:00 вшановуємо всіх загиблих на війні Українців.\n"
    "Сайт | Facebook | YouTube | ТikТok.",
    "Вічна пам'ять полеглим захисникам України..",
    "Вшануймо пам'ять та героїчний подвиг воїнів, полеглих під час захисту "
    "незалежності України.",
    # The digest, headed by the hour or the day rather than by the word.
    "Головне за день:",
    "Головне за день.",
    "Головне за день для киян:",
    "Головні новини за день:",
    "Головне на 10:00",
    "▶НАЖИВО Головне на 21:00",
    "ГОЛОВНЕ СТАНОМ НА 16:00",
    "🪓 Головне на 15:00 28 липня| .",
    "Головне від ЦПД за 27 липня 2026 року:",
    "🔵📣 Новини за цю ніч. Головне:",
    "📌 Головні новини дня 23.07.2026.",
    "ГОЛОВНІ НОВИНИ 27.07.2026",
    "🔷 Головні новини вівторка, 28 липня:",
    "5 головних новин за день: підсумок понеділка, 27 липня.",
    "Що трапилось за ніч:",
    "Що трапилося за день:",
    "🌙 ПІДСУМКИ ДНЯ | 27 липня.",
    "Підсумки тижня IT ARMY Kit (21–27 лип.)",
    # The digest that says the word, in the forms the patterns kept missing: as
    # the second word, and in the locative.
    "☀️ ДЕННИЙ ДАЙДЖЕСТ | 28 липня.",
    "Соціальний дайджест: причини відмови в пенсіях, небезпека сонячних панелей "
    "та сімейні маршрути.",
    "📌Дайджест за 28 липня — 1616-й день повномасштабного вторгнення Росії.",
    "У дайджесті вівторка – зустріч Трампа з Зеленським, арешт нападників у "
    "Вроцлаві, план реформ від ЄС.",
    # The topics first and the heading after them, which is how a digest lede is
    # written when the channel wants the first line to sell the post.
    "Трамп про зустріч із Зеленським, пожежа на Wildberries у Росії. "
    "Головне станом на ранок:\nПрезидент США заявив, що його зустріч із "
    "президентом України 28 липня у Білому домі «пройшла дуже добре».",
    "ФРОНТ | Ситуація на ранок, 29 липня.\nЗа оцінкою Інституту вивчення війни "
    "(ISW, США)",
    # The place ends with a period, and the period is what the alert patterns
    # tripped over: an abbreviated city, and a district on a line of its own.
    "❗️м. Київ — повітряна тривога!",
    "Ізмаїльський район.\nПовітряна тривога!",
    # And two districts before it, which is how the same channel writes an alert
    # covering both. Each line ends with a period, so the repeat has to be there.
    "Білгород-Дністровський район.\nІзмаїльський район.\nПовітряна тривога!",
    # Guided bombs. The air force names them the way it names drones, and the
    # munition list simply had no word for them.
    "КАБи на Дніпропетровщину.",
    "КАБи на Харківщину!",
    "КАБи на Запорізьку область, в напрямку обласного центру.",
    # A calendar rubric: what today is a holiday of.
    "27 липня: Яке сьогодні свято, все про цей день.",
]

# Digests that name themselves nowhere: no heading, no keyword, nothing but the
# shape. Four sibling lines, each a whole news item, each linked to the post that
# carried it — which is what `listing` reads instead of the words. The number is
# links to the channel's own earlier posts, not links in the post.
LISTINGS = [
    (
        "➡️ Зеленський прибув до США: зустріч із Дональдом Трампом, запрошення "
        "всіх 100 сенаторів США та обговорення санкцій проти Росії.\n"
        "➡️ WSJ: Дональд Трамп почав сприймати Володимира Зеленського як "
        "переможця, а його ставлення до Володимира Путіна погіршилося.\n"
        "➡️ Після зустрічі Зеленського і Трампа Україна та США домовилися "
        "інтенсифікувати переговори на всіх рівнях.\n"
        "➡️ Нацполіція викрила масштабну криптошахрайську схему: майже 1000 "
        "потерпілих і понад $1,1 млн збитків.",
        4,
    ),
    (
        "🔷 Зеленський обговорив із Трампом виробництво перехоплювачів для "
        "Patriot та подальші постачання систем ППО.\n"
        "🔷 Пентагон хоче переглянути умови передачі озброєння партнерам, "
        "повідомляє Reuters із посиланням на джерела.\n"
        "🔷 Україна отримала другий транш від МВФ майже на $690 млн за новою "
        "програмою розширеного фінансування.\n"
        "🔷 Федоров: закупівлі дронів наступного року зростуть щонайменше "
        "вдвічі порівняно з цьогорічними.\n"
        "🔷 У Росії готують умови для розширення мобілізації та планують "
        "залучити тридцять тисяч військових.",
        5,
    ),
]

# The same shape, and every one of them a single story rather than a list of
# them. The sub-points of an analysis, the counts in a police report, the
# quotations in a wire story: many marked lines, and almost no links, because
# there is one subject and nothing to link each point to.
NOT_LISTINGS = [
    (
        "Британский Королевский объединенный институт оборонных исследований "
        "опубликовал доклад о войне.\n"
        "• Россия потеряла инициативу на большинстве направлений.\n"
        "• Логистика по железной дороге деградировала.\n"
        "• Производство ракет выросло, но не покрывает расход.\n"
        "• Мобилизационный резерв сокращается быстрее прогнозов.",
        2,
    ),
    (
        "Хотів купити позитивний висновок НАЗК за $30 тисяч: ДБР затримало "
        "посадовця.\n"
        "🔹 Слідство встановило, що чоловік діяв через посередника.\n"
        "🔹 Гроші передавали двома частинами у Києві.\n"
        "🔹 Затриманому інкримінують пропозицію неправомірної вигоди.\n"
        "🔹 Санкція статті передбачає до восьми років позбавлення волі.",
        3,
    ),
    # Found by running the rule over six days of live posts and reading what it
    # newly caught. Every one of these is a news story whose paragraphs open the
    # way paragraphs do, and the first version of the rule counted that opening
    # as a bullet. They are the reason markers are an exclusion and not a guess.
    (
        "Удар росії по АТБ в Чернігові: воєнний злочин скоїли окупанти зі складу "
        "окремої бригади безпілотних систем.\n"
        "«Ми встановили підрозділ, який завдав удару по супермаркету», — сказав "
        "речник обласної прокуратури.\n"
        "«Йдеться про оператора, який керував дроном із тимчасово окупованої "
        "території», — додав він.\n"
        "«Слідство продовжує встановлювати всіх причетних до цього злочину», — "
        "зазначив прокурор у коментарі.",
        22,
    ),
    (
        "Фейкова співбесіда, «зламаний мікрофон» і порожній криптогаманець: нова "
        "хвиля атак на шукачів роботи.\n"
        "— Спершу жертві пишуть від імені відомої компанії й пропонують вакансію "
        "з великою зарплатою.\n"
        "— Потім просять встановити застосунок для відеозвʼязку, який нібито "
        "полагодить мікрофон.\n"
        "— Насправді це стилер, який вигрібає ключі від гаманців і паролі з "
        "браузера.",
        6,
    ),
    (
        # An analysis of one subject, with four bullets that are facets of it and
        # four links to the channel's own earlier pieces about it. Everything the
        # rule asks for except that the post is mostly prose: a title, an
        # opening, and a closing paragraph around the list.
        "❤️ 11 днів після звільнення Михайла Федорова: чи має він шанс "
        "повернутись?\n"
        "Забігаючи наперед скажу, що зараз жодних сигналів про можливе "
        "повернення Федорова до Міноборони НЕмає.\n"
        "🌟Зеленський у свіжому інтервʼю заявив, що між військовим командуванням "
        "і міністерством місяцями не було нормальної взаємодії.\n"
        "🌟Президент додав, що поважає Федорова, але «важливі операції — не "
        "справа однієї людини».\n"
        "🌟Зеленський кілька разів пропонував Федорову інші посади, такі як "
        "віцепремʼєр з військових новацій.\n"
        "🌟Федоров продовжує твердо наполягати, що погодиться лише на посаду "
        "міністра оборони України.\n"
        "Тим часом по всій країні продовжуються протести на підтримку Федорова, "
        "але їхня інтенсивність поступово спадає.",
        4,
    ),
    (
        "‍Аналіз майже восьми тисяч дописів у фейсбуці показав, як пишуть "
        "про переселенців і біженців у соцмережах.\n"
        "‍Найбільше публікацій — про житло та виплати, і саме вони "
        "збирають найбільше агресивних комментарів.\n"
        "‍Дослідники окремо рахували дописи, де переселенців згадують у "
        "звʼязку зі злочинами.\n"
        "‍За рік частка нейтральних згадок зросла, а частка ворожих "
        "залишилася на місці.",
        9,
    ),
]

# News. Every one of these is about the same subjects — the minute of silence,
# a funeral, the dead — and every one has to survive.
NEWS = [
    "У Києві сьогодні вперше перекрили вулицю Хрещатик о 9 ранку на час "
    "загальнонаціональної хвилини мовчання. Тепер так буде щодня.",
    "Володимир Зеленський підписав указ про загальнонаціональну хвилину "
    "мовчання «Щоранку о 9-й годині на всій території нашої держави будемо "
    "згадувати усіх, кого забрала ця війна», – йдеться в указі президента.",
    "У метро Києва щодня о 09:00 оголошуватимуть загальнонаціональну хвилину "
    "мовчання. Відсьогодні оголошення лунатиме на 19 станціях, де це технічно "
    "можливо. Згодом ці заходи запровадять і в салонах автобусів.",
    "Щоранку рівно о 9 годині у Вінниці, як і в інших містах України "
    "оголошують загальнонаціональну хвилину мовчання. Таким чином вшановують "
    "пам’ять усіх загиблих. Мерія повідомила, що сигнал лунатиме на Хрещатику "
    "та Майдані, його показуватимуть на рекламних носіях і в застосунку "
    "«Київ Цифровий». Закладам рекомендували зупиняти обслуговування.",
    "У громадському транспорті Києва оголошуватимуть хвилину мовчання. "
    "О 9:00 сигнал лунатиме на Хрещатику та Майдані Незалежності, його "
    "показуватимуть на рекламних носіях і в застосунку «Київ Цифровий».",
    "За відмову слухати гімн штрафуватимуть? Петиція з'явилася на сайті "
    "Кабміну. У ній пропонують після «хвилини мовчання» вмикати Гімн, а за "
    "публічне ігнорування притягати до адміністративної відповідальності.",
    "Росія вбиває. Щодня. Російський терор не знає пауз. На Київщині "
    "російські ракети забрали життя щонайменше шістьох людей. Десятки "
    "отримали поранення. Висловлюю щирі співчуття родинам загиблих.",
    "Коли Росія вбиває дітей — дипломатія вже не працює. Росія завдала удару "
    "по супермаркету в Чернігові. Місцю, куди люди у вихідний прийшли за "
    "звичними покупками. Щирі співчуття всім, хто втратив близьких.",
    "Ігор Клименко повідомив, що по прибуттю на місце вбивства Ірини Фаріон "
    "поліцейські вилучили гільзу калібром 9х18 мм. Наразі вона на експертному "
    "дослідженні. «Зброя специфічна», — зазначив міністр.",
    "Президент України Володимир Зеленський разом з дружиною Оленою взяли "
    "участь у церемонії прощання з академіком, Героєм України Борисом "
    "Патоном. Прощання з видатним науковцем відбулося у Києві.",
    "У Косові на Франківщині будуть закривати громадські заклади під час "
    "прощання з полеглими військовими. Таке рішення прийняла Косівська "
    "районна військова адміністрація.",
    "Blizzard перервала мовчання і анонсувала показ Overwatch 2. Трансляція "
    "буде присвячена PvP-режиму гри, і триватиме 2 години.",
    "Нардеп Мовчан розповів про можливості законопроекту №4572 для "
    "прискорення приватизації та залучення інвестицій.",
    "Супер факти про собак-рятувальників. Собаки працюють рятувальниками вже "
    "більше трьохсот років. Першими їх допомогу почали використовувати монахи "
    "з високогірних притулків. Вони знаходили людей під снігом і проводжали "
    "загиблих альпіністів в останню путь, якщо було вже надто пізно. Сьогодні "
    "кінологи ДСНС тренують собак за міжнародними стандартами.",
    # News about the air-raid stream rather than a part of it: a rule changes, a
    # strike happens, someone counts the sirens. The distinction is the verb —
    # an alert post states that the siren is on, these say what came of it.
    "Уряд змінює систему оповіщення про повітряну тривогу: сирени доповнить "
    "мобільний застосунок, повідомив Кабмін.",
    "Школи Києва переходять на дистанційне навчання через частіші повітряні "
    "тривоги — рішення ухвалила міськрада.",
    "Під час повітряної тривоги в Харкові ракета влучила в багатоповерхівку. "
    "Загинули п'ятеро людей, ще дванадцять постраждали.",
    "Мер закликав киян не нехтувати сигналами тривоги і прямувати в укриття, "
    "адже кількість жертв серед тих, хто залишається на вулиці, зростає.",
    "Повітряні сили попередили про загрозу балістики для східних областей — це "
    "вже третій зліт МіГ-31К за тиждень.",
    "Тривога в суспільстві зростає: за даними опитування КМІС, 62% українців "
    "відчувають постійний стрес через обстріли.",
    "У Києві відкрили 50 нових укриттів, куди можна прямувати в укриття під час "
    "тривоги — мерія оприлюднила карту.",
    "Мерія розповіла, чому в Києві оголосили тривогу лише через десять хвилин "
    "після пусків. Причиною назвали збій у системі передачі даних від "
    "військових, який уже виправили. Депутати вимагають перевірки.",
    # "Курсом на" is a headline idiom as often as it is a drone's heading.
    "Курсом на Європу: Україна отримала статус кандидата у члени ЄС",
    "Кількість повітряних тривог у липні зросла втричі порівняно з червнем, "
    "свідчать дані «Тривога.Онлайн».",
    # From the same 262 live posts as the block at the end of ROUTINE. These are
    # what the patterns broadened for that block have to keep their hands off:
    # the reporting the same channels publish between the alerts.
    "Упродовж минулого тижня ворожих ударів зазнали 77 населених пунктів "
    "Харківської області, зокрема м. Харків. Внаслідок обстрілів постраждали "
    "115 людей, серед них — 7 дітей.",
    "Сили ППО знешкодили 107 з 131 ворожих дронів. Зафіксовано влучання 16 "
    "ударних БпЛА на 9 локаціях, а також падіння збитих (уламки) на 6 локаціях. "
    "Атака триває.",
    "Володимир Зеленський уже прибув до США з одноденним візитом. Головний "
    "пріоритет поїздки, як заявив президент, — посилення протиповітряної "
    "оборони, зокрема антибалістики, і стратегічна співпраця.",
    "У Києві з 28 липня до 2 серпня триватимуть сезонні ярмарки. Локації – на "
    "сайті kyivcity.gov.ua. Закликаємо відвідувачів пам'ятати про власну "
    "безпеку та під час повітряної тривоги перейти в укриття.",
    "ЗБИТО/ПОДАВЛЕНО 107 ЦІЛЕЙ\nУ ніч на 28 липня (з 18:00 27 липня) противник "
    "атакував 131 ударним БпЛА типу Shahed (в т.ч. реактивними), Гербера, "
    "Італмас та дронами-імітаторами типу «Пародія».",
    "Протягом цього тижня на території області сигнал «Повітряна тривога» не "
    "оголошувався. Разом із начальником Захисту відпрацювали план на серпень.",
    # From the same six days as the block at the end of ROUTINE, and the reason
    # that block is worded as narrowly as it is. Each of these repeats daily too,
    # which is exactly why recurrence cannot be allowed to decide on its own:
    # the general staff's situation report and the grid status are the routine of
    # a channel whose routine is the news.
    "Оперативна інформація станом на 16:00 23.07.2026 щодо російського "
    "вторгнення. Протягом доби відбулося 180 боєзіткнень.",
    "Загальні бойові втрати противника з 24.02.22 по 24.07.26 (орієнтовно): "
    "особового складу — близько 1042060 осіб.",
    "СТАН ЕНЕРГОСИСТЕМИ.\nСпоживання в межах прогнозу, обмежень не "
    "застосовували.",
    "Добовий звіт (25.07.26) результатів угруповання СБС.",
    "Армія РФ за добу втратила 1560 солдатів.",
    "За добу Сили безпілотних систем уразили або знищили 1761 ціль противника.",
    "На фронті відбулось 180 боїв за добу, найбільше на покровському та "
    "костянтинівському напрямках – Генштаб.",
    "За добу через російські атаки в Україні загинули щонайменше дев'ять людей, "
    "понад 55 — поранені: у Краматорську та Слов'янську пошкоджено житло.",
    "За ніч ППО збито/подавлено 107 цілей зі 131.",
    # "Головне" as part of a headline about a subject, not a time window. The
    # difference is a single preposition, and it is the whole difference.
    "Велика Британія надасть Україні нові технології для захисту дронів — "
    "головне зі заяв Зеленського і Бернема:",
    "Головне про новий закон про мобілізацію: кого це торкнеться і коли.",
    "Головне на фронті: ЗСУ звільнили Андріївку на Донеччині, повідомив Генштаб.",
    "Головні новини на ринку нерухомості: ціни в Києві зросли на 8% за рік.",
    # "За добу" and "ситуація" in ordinary reporting.
    "До +2 грн на стелах: за добу пальне на українських АЗС знову подорожчало.",
    "Відеокарти подорожчали — ціни зросли до 30% за добу.",
    "За добу на Миколаївщині сталося 40 пожеж, більшість — через обстріли.",
    "У Львові гримить, ситуація в місті спокійна.",
    # Two sentences, the second of which starts with the siren. The place-first
    # patterns were widened to tolerate a period, and this is the post that
    # widening must not reach: the period ends a sentence, not a place name.
    "Росіяни атакували Київ. Повітряна тривога триває вже годину, у місті "
    "працює ППО.",
    # The munition list learned the word "КАБ", which begins the government's
    # name and half a dozen ordinary ones.
    "Кабмін затвердив нові правила бронювання працівників на серпень.",
    "Кабінет міністрів у вівторок ухвалив постанову про виплати переселенцям.",
    "Кабельні мережі на Харківщині відновили після обстрілу.",
]


@pytest.mark.parametrize("text", ROUTINE, ids=range(len(ROUTINE)))
def test_routine_posts_are_recognised(detector: RubricDetector, text: str) -> None:
    assert detector(text), f"missed: {' '.join(text.split())[:70]}"


@pytest.mark.parametrize(("text", "links"), LISTINGS, ids=range(len(LISTINGS)))
def test_a_digest_with_no_heading_is_recognised_by_its_shape(
    detector: RubricDetector, text: str, links: int
) -> None:
    assert detector.listing(text, links), f"missed: {' '.join(text.split())[:70]}"


@pytest.mark.parametrize(("text", "links"), NOT_LISTINGS, ids=range(len(NOT_LISTINGS)))
def test_one_story_with_sub_points_is_not_a_listing(
    detector: RubricDetector, text: str, links: int
) -> None:
    matched = detector.listing(text, links)
    assert not matched, f"{matched!r} wrongly caught: {' '.join(text.split())[:70]}"


def test_the_shape_alone_is_not_enough_without_the_own_links(
    detector: RubricDetector,
) -> None:
    """The two halves of the rule, separated.

    Marked lines say the post is a list; links into the channel's own history say
    each item is a story of its own. The first listing above, offered without
    those links, has to survive — which is what keeps an enumerated report of one
    night's strikes out of this rule.
    """
    text, links = LISTINGS[0]

    assert detector.listing(text, links)
    assert not detector.listing(text, 0)


def test_a_listing_needs_four_items(detector: RubricDetector) -> None:
    """Three is where longform articles citing their own earlier pieces live."""
    text = (
        "➡️ Перший фігурант справи визнав вину у суді Києва цього тижня.\n"
        "➡️ Другий фігурант переховується за кордоном, оголошено розшук.\n"
        "➡️ Третій фігурант вийшов під заставу і вже покинув територію країни."
    )

    assert not detector.listing(text, 5)


def test_the_shape_rule_is_separate_from_the_wording_rules(
    detector: RubricDetector,
) -> None:
    """`__call__` stays about text alone, so a caller holding only text is right."""
    assert detector("Дайджест новин: окупанти вдарили по Миколаєву.")
    assert not detector(LISTINGS[0][0])


@pytest.mark.parametrize("text", NEWS, ids=range(len(NEWS)))
def test_news_about_the_same_subjects_survives(
    detector: RubricDetector, text: str
) -> None:
    matched = detector.explain(text)
    assert not matched, (
        f"pattern {matched!r} wrongly caught: {' '.join(text.split())[:70]}"
    )


def test_a_detector_with_no_patterns_catches_nothing() -> None:
    detector = RubricDetector({})

    assert not detector("Щоранку о 9:00 – хвилина мовчання")


def test_a_detector_with_no_patterns_says_so(caplog: pytest.LogCaptureFixture) -> None:
    """The one failure mode that leaves no trace anywhere else.

    `configs` is a mounted volume, so a deploy that ships new code over an
    unchanged config file gets a config with no `rubric_detector` section at
    all, and the detector is then built with nothing to match. Every ritual
    post stays in the feed and nothing is logged: the siren, the all-clear and
    the minute of silence went on being published for a day this way.
    """
    with caplog.at_level(logging.WARNING):
        RubricDetector({})

    assert "no patterns" in caplog.text


def test_own_posts_are_counted_and_the_footer_is_not() -> None:
    """The distinction the shape rule rests on.

    Ukrinform signs every post with links to its own front page and its social
    accounts. Counting those made an enumerated report of one night's strikes
    look like a digest of four separate stories, so only a link to a numbered
    post of this same channel counts.
    """
    links = [
        "https://t.me/kaptuz/69174",
        "https://t.me/kaptuz/69178/",
        "https://t.me/kaptuz",  # the front page: every sign-off has one
        "https://t.me/othernews/1234",  # someone else's post
        "https://facebook.com/kaptuz",
        "https://tsn.ua/ato/some-story-3137726.html",
    ]

    assert count_own_posts(links, "kaptuz") == 2


def test_own_posts_tolerate_the_channel_id_as_written() -> None:
    """`channel_id` reaches this with the case and the @ it was written with."""
    links = ["https://t.me/Kaptuz/69174", "https://T.ME/kaptuz/69178"]

    assert count_own_posts(links, "@KaptuZ") == 2


def test_a_channel_without_an_id_counts_nothing() -> None:
    """A document whose channel never resolved must not match every link."""
    assert count_own_posts(["https://t.me/kaptuz/69174"], "") == 0


def test_empty_text_is_not_a_rubric(detector: RubricDetector) -> None:
    assert not detector("")
    assert not detector("   \n  ")


def test_explain_names_the_pattern_that_matched(detector: RubricDetector) -> None:
    """A post vanishing from the feed should be traceable to one line of config."""
    matched = detector.explain("Дайджест новин: окупанти вдарили по Миколаєву.")

    assert matched is not None
    assert "дайджест" in matched


def test_explain_answers_with_the_line_as_written(detector: RubricDetector) -> None:
    """Macros stay unexpanded: the point is to name the line a human must edit."""
    matched = detector.explain("🔴 Повітряна тривога в Києві")

    assert matched is not None
    assert "%not_a_strike%" in matched


@pytest.mark.parametrize(
    "text",
    [
        "- " * 2048,
        "—" * 4096,
        "УВАГА " + "- " * 2000,
        "\n" * 4096 + "відбій",
        "УВАГА! " + ("Район " * 10 + "\n") * 60 + "ВІДБІЙ",
        ("а" * 79 + " - ") * 50,
        # For the branch that lets a place name end with a period: the same
        # bounded class, now inside a repeat of its own.
        ("Район. " * 8 + "\n") * 60 + "ВІДБІЙ",
        ("м. " + "а" * 58 + ".\n") * 80 + "ПОВІТРЯНА ТРИВОГА",
    ],
    ids=range(8),
)
def test_a_pathological_post_does_not_hang_the_detector(
    detector: RubricDetector, text: str
) -> None:
    """The place prefix is optional, bounded and next to other optional parts.

    Written the obvious way — `(?:[^.!?,]{0,50}(?:[—–:-]|\\n)\\s*)*`, a bounded
    class inside an unbounded repeat — it took over five seconds on a post of
    four thousand dashes, which is inside Telegram's 4096-character limit and so
    is a post a channel can actually publish. The bounds on the `\\W` runs are
    what keeps this linear; they are not decoration.
    """
    start = time.perf_counter()
    detector(text)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"took {elapsed:.2f}s on {len(text)} characters"


def test_an_unknown_macro_is_an_error() -> None:
    """A typo must fail loudly, not compile to a pattern that never matches."""
    with pytest.raises(KeyError, match="typo"):
        RubricDetector({"patterns": ["^%typo%тривога"], "macros": {}})
