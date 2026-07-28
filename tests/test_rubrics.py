import json
import logging
import time

import pytest

from nyan.rubrics import RubricDetector

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
]


@pytest.mark.parametrize("text", ROUTINE, ids=range(len(ROUTINE)))
def test_routine_posts_are_recognised(detector: RubricDetector, text: str) -> None:
    assert detector(text), f"missed: {' '.join(text.split())[:70]}"


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
    ],
    ids=range(6),
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
