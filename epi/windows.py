"""Нарезка каталога на обучающую выборку из окон одинаковой формы.

    from epi import Catalog, WindowSet

    windows = WindowSet(Catalog("catalog.sqlite"))
    windows.build()                       # разметить окна
    windows.split()                       # раздать пациентов по фолдам
    x, y = windows.sample("train", n=512)  # массивы прямо в память

Три свойства этих данных определяют устройство модуля:

* запись состоит из сегментов, разделённых настоящими разрывами во времени,
  поэтому окно обязано лежать внутри одного сегмента и не пересекать границу;
* усечённые выгрузки добиты сплошными нулями, поэтому окно нужно проверять на
  плоский сигнал, прежде чем брать его в выборку;
* монтажи у записей разные, поэтому окна урезаются до 19 электродов схемы
  10-20, которые есть в каждой записи, и приводятся к общей частоте.

Разбиение делается по пациентам. Окна одной записи перекрываются во времени и
тривиально предсказываются друг из друга, поэтому случайное разбиение самих
окон протащило бы тестовую выборку в обучение.

Каждое окно знает, насколько оно отстоит от ближайшего приступа: lead_sec --
секунды до начала следующего приступа, lag_sec -- секунды после конца
предыдущего. От этого расстояния зависит метка, и им же настраивается ширина
промежутка между окном и приступом:

    WindowSet(catalog, gap_before_sec=60, gap_after_sec=300)

Окна ближе указанных зазоров не годятся ни в положительные, ни в фоновые:
перед приступом сигнал уже меняется, после него держится постиктальное
угнетение. Такие окна получают метку -1 и в выборку не идут. Зазоры заданы
раздельно, потому что постиктальный хвост длиннее преиктального.

Тот же зазор превращает задачу обнаружения в задачу прогноза:

    WindowSet(catalog, gap_before_sec=300, preictal_sec=1800)

Теперь положительны окна, отстоящие от начала приступа на 5-35 минут: за 5
минут прогноз ещё имеет смысл (успеть предупредить), а глубже 35 минут сигнал
уже неотличим от фона. Сам приступ и постиктальный хвост из выборки уходят.

Приступ в разметке -- не маркер врача, а событие:

    WindowSet(catalog, merge_gap_sec=120, positives_per_event=1)

Приступы, между которыми меньше merge_gap_sec секунд, склеиваются в одно
событие: короткая передышка между двумя разрядами -- не фон, и окна оттуда
нечего противопоставлять самим приступам. А positives_per_event ограничивает
вклад одного события в выборку: соседние окна одного приступа перекрываются и
отличаются сдвигом на шаг, так что десяток почти одинаковых примеров даёт
модели ровно столько же, сколько один, но весит в десять раз больше.

Расстояние до приступа управляет и тем, из чего набирается фон:

    windows.sample("train", n=512, hard_ratio=0.5, hard_span_sec=600)

Случайный фон почти весь приходит из спокойных часов, и модель учится отличать
приступ от сна, а не от того, что на него похоже. hard_ratio задаёт долю
фоновых окон, взятых вплотную к приступу -- из полосы шириной hard_span_sec
сразу за зазором, с обеих сторон.
"""

from collections import defaultdict
from fractions import Fraction
import logging
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .reader import NicoletEReader

logger = logging.getLogger(__name__)


class WindowReadError(RuntimeError):
    """Ридер не смог вернуть полное окно запрошенной формы."""


# 19 электродов схемы 10-20, общие для всех записей коллекции
CANONICAL = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
TARGET_RATE = 256.0
SEIZURE_TYPES = ("Seizure",)

# метка окна, попавшего в зазор около приступа: ни приступ, ни чистый фон
GREY = -1
USABLE = "label >= 0"

SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    id           INTEGER PRIMARY KEY,
    recording_id INTEGER,
    segment      INTEGER,
    start_sec    REAL,
    length_sec   REAL,
    label        INTEGER,
    overlap_sec  REAL,
    lead_sec     REAL,
    lag_sec      REAL,
    event_idx    INTEGER
);
CREATE TABLE IF NOT EXISTS splits (
    patient_key TEXT PRIMARY KEY,
    fold        TEXT
);
CREATE INDEX IF NOT EXISTS windows_by_label ON windows(label);
CREATE INDEX IF NOT EXISTS windows_by_recording ON windows(recording_id);
"""


def overlap(window_start, window_end, spans):
    """Сколько секунд окна попадает внутрь размеченных интервалов."""
    return sum(
        max(0.0, min(window_end, stop) - max(window_start, start))
        for start, stop in spans
    )


def merge(spans, gap_sec=0.0):
    """Слить интервалы, между которыми меньше gap_sec секунд, в один.

    При нулевом зазоре сливаются только пересекающиеся и стыкующиеся
    интервалы: два маркера на один и тот же приступ иначе посчитались бы
    дважды.
    """
    merged = []
    for start, stop in sorted(spans):
        if merged and start - merged[-1][1] <= gap_sec:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(start, stop) for start, stop in merged]


def locate(window_start, window_end, spans):
    """Где окно стоит относительно интервалов: (lead, lag, event).

    lead  -- секунды от конца окна до начала следующего интервала;
    lag   -- секунды от конца предыдущего интервала до начала окна;
    event -- номер интервала, к которому окно отнесено: того, что оно задевает,
             иначе ближайшего следующего, иначе ближайшего предыдущего.

    None означает, что с этой стороны интервалов нет вовсе (а у event -- что их
    нет во всём сегменте). Окно, задевшее интервал, стоит от него на нуле с
    обеих сторон.
    """
    lead = lag = event = None
    ahead_idx = behind_idx = None
    for index, (start, stop) in enumerate(spans):
        if start >= window_end:
            ahead = start - window_end
            if lead is None or ahead < lead:
                lead, ahead_idx = ahead, index
        elif stop <= window_start:
            behind = window_start - stop
            if lag is None or behind < lag:
                lag, behind_idx = behind, index
        else:
            lead = lag = 0.0
            event = index
    if event is None:
        event = ahead_idx if ahead_idx is not None else behind_idx
    return lead, lag, event


class WindowSet:
    """Окна фиксированной длины, разбиение по пациентам и выборка данных."""

    def __init__(self, catalog, length_sec=10.0, stride_sec=5.0,
                 min_overlap_sec=5.0, gap_before_sec=0.0, gap_after_sec=0.0,
                 preictal_sec=None, merge_gap_sec=0.0, positives_per_event=None,
                 rate=TARGET_RATE, channels=None):
        """catalog -- объект Catalog.

        length_sec          -- длина окна;
        stride_sec          -- шаг между началами соседних окон;
        min_overlap_sec     -- сколько секунд приступа должно попасть в окно,
                               чтобы оно считалось положительным;
        gap_before_sec      -- ширина зазора перед приступом: окна, отстоящие
                               от его начала меньше чем на столько секунд, в
                               выборку не идут;
        gap_after_sec       -- то же после конца приступа (постиктальный хвост);
        preictal_sec        -- длина окна ожидания приступа. Если задано, задача
                               меняется с обнаружения на прогноз: положительны
                               окна, отстоящие от начала приступа на
                               gap_before_sec ... gap_before_sec + preictal_sec,
                               а сам приступ уходит в исключённые;
        merge_gap_sec       -- приступы, между которыми меньше стольких секунд,
                               считаются одним событием;
        positives_per_event -- сколько положительных окон брать с одного
                               события; None -- сколько выйдет;
        rate                -- общая частота, к которой всё приводится;
        channels            -- список каналов; по умолчанию схема 10-20.
        """
        self.catalog = catalog
        self.conn = catalog.conn
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.length_sec = length_sec
        self.stride_sec = stride_sec
        self.min_overlap_sec = min_overlap_sec
        self.gap_before_sec = gap_before_sec
        self.gap_after_sec = gap_after_sec
        self.preictal_sec = preictal_sec
        self.merge_gap_sec = merge_gap_sec
        self.positives_per_event = positives_per_event
        self.rate = rate
        self.channels = list(channels) if channels else list(CANONICAL)
        self._readers = {}

    def _migrate(self):
        """Добить таблицу окон колонками, которых нет в каталогах прошлых версий."""
        have = {row["name"] for row in self.conn.execute("PRAGMA table_info(windows)")}
        for column, kind in (("lead_sec", "REAL"), ("lag_sec", "REAL"),
                             ("event_idx", "INTEGER")):
            if column not in have:
                self.conn.execute("ALTER TABLE windows ADD COLUMN %s %s"
                                  % (column, kind))
        self.conn.commit()

    def __repr__(self):
        row = self.conn.execute(
            """SELECT COUNT(*) n, SUM(label = 1) p FROM windows WHERE %s""" % USABLE
        ).fetchone()
        return "<WindowSet %.0f с: %d окон, %d %s>" % (
            self.length_sec, row["n"], row["p"] or 0, self.positive_name
        )

    @property
    def positive_name(self):
        """Как называется положительный класс при текущих настройках."""
        return "с приступом" if self.preictal_sec is None else "преиктальных"

    # ------------------------------------------------------------------
    # разметка

    def _seizure_spans(self, recording_id, segment):
        """Интервалы приступов одного сегмента в секундах от его начала."""
        spans = []
        for row in self.conn.execute(
            """SELECT segment_time_sec, duration_sec FROM events
               WHERE recording_id = ? AND segment = ? AND type IN (%s)"""
            % ",".join("?" * len(SEIZURE_TYPES)),
            (recording_id, segment) + SEIZURE_TYPES,
        ):
            start = row["segment_time_sec"]
            spans.append((start, start + (row["duration_sec"] or 0.0)))
        return spans

    def _label(self, hit, lead, lag):
        """Метка окна: 1 -- положительное, 0 -- фон, -1 -- зазор около приступа.

        hit  -- сколько секунд приступа попало в окно;
        lead -- секунды до начала следующего приступа, None если его нет;
        lag  -- секунды после конца предыдущего, None если его нет.

        Зазор проверяется раньше фона, поэтому нулевые gap_before_sec и
        gap_after_sec оставляют разметку ровно такой, какой она была без них.
        """
        in_gap = ((lead is not None and lead < self.gap_before_sec)
                  or (lag is not None and lag < self.gap_after_sec))

        if self.preictal_sec is None:
            if hit >= self.min_overlap_sec:
                return 1
            return GREY if in_gap else 0

        # прогноз: сам приступ и его окрестности -- не пример ни того, ни
        # другого класса, положительно только окно ожидания перед приступом
        if hit > 0 or in_gap:
            return GREY
        if lead is not None and lead <= self.gap_before_sec + self.preictal_sec:
            return 1
        return 0

    def _thin(self, made):
        """Оставить с каждого события не больше positives_per_event окон.

        Соседние окна одного приступа перекрываются и отличаются сдвигом на шаг
        -- как примеры они почти дубликаты, а вес события в выборке раздувают.
        Лишние уходят в исключённые, а не в фон: приступ в них всё-таки есть.
        Заодно исключается всё, что событие лишь задело.
        """
        if self.positives_per_event is None:
            return made

        by_event = defaultdict(list)
        for row in made:
            if row[4] == 1:
                by_event[row[8]].append(row)
        for group in by_event.values():
            # представителем события идёт окно, набравшее больше всего приступа,
            # а при прогнозе -- ближайшее к его началу
            group.sort(key=lambda row: (-row[5], row[6] if row[6] is not None else 0.0,
                                        row[2]))
            for row in group[self.positives_per_event:]:
                row[4] = GREY
        for row in made:
            if row[4] == 0 and row[5] > 0:
                row[4] = GREY
        return made

    def build(self, verbose=True):
        """Перебрать сегменты, нарезать их на окна и разметить приступы.

        Приступы, между которыми меньше merge_gap_sec секунд, идут в разметку
        одним событием: промежуток между двумя близкими приступами -- не фон, и
        противопоставлять окна оттуда самим приступам бессмысленно.

        Возвращает словарь: total, usable, positive, excluded, seizures, events,
        positive_share, skipped_segments. Доля положительных считается от годных
        окон, то есть без тех, что попали в зазор около приступа.
        """
        self.conn.execute("DELETE FROM windows")
        rows, seizures, events = [], 0, 0
        for segment in self.conn.execute(
            """SELECT segments.*, recordings.id AS rec FROM segments
               JOIN recordings ON recordings.id = segments.recording_id
               WHERE recordings.deleted_at IS NULL"""
        ).fetchall():
            marked = self._seizure_spans(segment["rec"], segment["idx"])
            spans = merge(marked, self.merge_gap_sec)
            seizures += len(marked)
            events += len(spans)
            made = []
            start = 0.0
            while start + self.length_sec <= segment["duration_sec"]:
                stop = start + self.length_sec
                hit = overlap(start, stop, spans)
                lead, lag, event = locate(start, stop, spans)
                made.append([segment["rec"], segment["idx"], start, self.length_sec,
                             self._label(hit, lead, lag), hit, lead, lag, event])
                start += self.stride_sec
            rows.extend(self._thin(made))

        self.conn.executemany(
            """INSERT INTO windows (recording_id, segment, start_sec, length_sec,
                                    label, overlap_sec, lead_sec, lag_sec, event_idx)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows,
        )
        self.conn.commit()

        positive = sum(r[4] == 1 for r in rows)
        excluded = sum(r[4] == GREY for r in rows)
        usable = len(rows) - excluded
        skipped = self.conn.execute(
            """SELECT COUNT(*) n FROM segments JOIN recordings
               ON recordings.id = segments.recording_id
               WHERE recordings.deleted_at IS NULL AND segments.duration_sec < ?""",
            (self.length_sec,),
        ).fetchone()["n"]
        report = {
            "total": len(rows),
            "usable": usable,
            "positive": positive,
            "excluded": excluded,
            "seizures": seizures,
            "events": events,
            "positive_share": 100 * positive / max(usable, 1),
            "skipped_segments": skipped,
        }
        if verbose:
            print("окна по %.0f с с шагом %.0f с: %d всего, %d %s (%.2f%%)"
                  % (self.length_sec, self.stride_sec, report["usable"],
                     report["positive"], self.positive_name,
                     report["positive_share"]))
            if seizures != events:
                print("приступов %d, после объединения по %.0f с событий %d"
                      % (seizures, self.merge_gap_sec, events))
            if excluded:
                print("исключено окон: %d%s" % (excluded, self._why_excluded()))
            print("сегментов короче одного окна: %d" % skipped)
        return report

    def _why_excluded(self):
        """Перечислить настройки, из-за которых окна уходят в исключённые."""
        reasons = []
        if self.gap_before_sec or self.gap_after_sec:
            reasons.append("зазор %.0f с до и %.0f с после приступа"
                           % (self.gap_before_sec, self.gap_after_sec))
        if self.preictal_sec is not None:
            reasons.append("сам приступ")
        if self.positives_per_event is not None:
            reasons.append("с события не больше %d" % self.positives_per_event)
        return (" (%s)" % ", ".join(reasons)) if reasons else ""

    def split(self, seed=0, holdout=0.15, verbose=True):
        """Раздать пациентов целиком по фолдам, разнося носителей приступов.

        Положительные окна сосредоточены у нескольких пациентов, поэтому
        случайное групповое разбиение легко оставляет фолд вовсе без приступов.
        Пациенты раздаются от самых богатых приступами к бедным, и каждый
        уходит в тот фолд, который сильнее прочих отстаёт от своей доли.

        Возвращает список словарей по фолдам.
        """
        patients = self.conn.execute(
            """SELECT patient_key,
                      SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS positives,
                      COUNT(*) AS windows
               FROM windows
               JOIN recordings ON recordings.id = windows.recording_id
               WHERE %s
               GROUP BY patient_key""" % USABLE
        ).fetchall()

        generator = np.random.default_rng(seed)
        order = generator.permutation(len(patients))
        ranked = sorted(
            ((patients[i], int(order[i])) for i in range(len(patients))),
            key=lambda item: (-item[0]["positives"], item[1]),
        )

        targets = {"train": 1 - 2 * holdout, "val": holdout, "test": holdout}
        have = {fold: 0.0 for fold in targets}
        assignment = {}
        for row, _ in ranked:
            total = sum(have.values()) + max(row["positives"], 1)
            fold = min(targets, key=lambda f: have[f] - targets[f] * total)
            assignment[row["patient_key"]] = fold
            have[fold] += max(row["positives"], 1)

        self.conn.execute("DELETE FROM splits")
        self.conn.executemany("INSERT INTO splits VALUES (?, ?)", assignment.items())
        self.conn.commit()

        report = self.folds()
        if verbose:
            for row in report:
                print("%-6s %2d пациентов, %6d окон, %5d %s"
                      % (row["fold"], row["patients"], row["windows"],
                         row["positive"], self.positive_name))
        return report

    def folds(self):
        """Размер каждого фолда: пациенты, окна, положительные окна.

        Окна из зазора около приступа не в счёт: в обучение они не попадут.
        """
        return self.catalog.query(
            """SELECT fold, COUNT(DISTINCT patient_key) AS patients,
                      COUNT(*) AS windows, COALESCE(SUM(label = 1), 0) AS positive
               FROM windows
               JOIN recordings ON recordings.id = windows.recording_id
               JOIN splits USING (patient_key)
               WHERE %s
               GROUP BY fold ORDER BY windows DESC""" % USABLE
        )

    def windows(self, fold=None, label=None, limit=None):
        """Строки окон с фильтрами по фолду и метке.

        По умолчанию отдаются только годные окна. Чтобы посмотреть на те, что
        отброшены зазором около приступа, нужно спросить их прямо: label=-1.
        """
        where, params = [USABLE if label is None else "label = ?"], []
        if label is not None:
            params.append(label)
        if fold is not None:
            where.append("fold = ?")
            params.append(fold)
        clause = "WHERE " + " AND ".join(where)
        sql = ("""SELECT windows.*, recordings.rel_path, recordings.patient_key
                  FROM windows
                  JOIN recordings ON recordings.id = windows.recording_id
                  LEFT JOIN splits USING (patient_key) %s""" % clause)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self.catalog.query(sql, params)

    # ------------------------------------------------------------------
    # чтение данных

    def _reader(self, recording_id):
        """Открытые ридеры кешируются: разбор заголовка стоит дороже чтения окна."""
        if recording_id not in self._readers:
            self._readers[recording_id] = NicoletEReader(
                self.catalog.path_of(recording_id), type_names={}
            )
        return self._readers[recording_id]

    def load(self, window):
        """Прочитать одно окно как массив (каналы, отсчёты) типа float32."""
        reader = self._reader(window["recording_id"])
        reader.select_segment(window["segment"])
        try:
            data = reader.read(self.channels, start_sec=window["start_sec"],
                               duration_sec=window["length_sec"])
        except (IndexError, OSError, EOFError, ValueError) as error:
            # ValueError бывает и при усечённом блоке: pynicolet пытается
            # записать меньше отсчётов, чем ожидалось, в свой выходной массив.
            raise WindowReadError(str(error)) from error
        if data.ndim != 2 or data.shape[1] != len(self.channels):
            raise WindowReadError(
                "expected (samples, %d channels), got %s"
                % (len(self.channels), data.shape)
            )
        expected_source = int(round(window["length_sec"] * reader.sampling_rate))
        if data.shape[0] < expected_source or data.shape[0] == 0:
            raise WindowReadError(
                "incomplete window: expected %d samples at %g Hz, got %d"
                % (expected_source, reader.sampling_rate, data.shape[0])
            )
        if reader.sampling_rate != self.rate:
            # дробь из двух частот, а не из их частного: частное -- двоичное
            # число с плавающей точкой, точное отношение которого имеет
            # астрономический знаменатель
            ratio = Fraction(int(self.rate), int(reader.sampling_rate))
            data = resample_poly(data, ratio.numerator, ratio.denominator, axis=0)
        expected = int(round(window["length_sec"] * self.rate))
        if data.shape[0] < expected:
            raise WindowReadError(
                "incomplete window: expected %d samples at %g Hz, got %d"
                % (expected, self.rate, data.shape[0])
            )
        return data[:expected].T.astype(np.float32)

    @staticmethod
    def flat_fraction(data):
        """Доля отсчётов, где все электроды показывают ровно ноль."""
        return float(np.mean(np.all(data == 0, axis=0)))

    # окно рядом с приступом: зазор оно уже пережило, но лежит вплотную к нему
    NEAR = """((lead_sec IS NOT NULL AND lead_sec <= ?)
               OR (lag_sec IS NOT NULL AND lag_sec <= ?))"""

    def sample(self, fold="train", n=1000, positive_ratio=0.5, hard_ratio=0.0,
               hard_span_sec=600.0, seed=0, max_flat=0.01, verbose=True,
               return_groups=False):
        """Собрать сбалансированную выборку окон в память.

        Классы набираются по отдельности, каждый до своей квоты, иначе редкие
        приступы утонули бы в случайной выборке. Окна с долей плоского сигнала
        выше max_flat отбрасываются, окна из зазора около приступа не
        рассматриваются вовсе.

        hard_ratio     -- какая доля фоновых окон берётся из трудных: тех, что
                          стоят к приступу вплотную, сразу за зазором. Случайный
                          фон почти весь набирается из спокойных часов, и модель
                          учится отличать приступ от сна, а не от того, что на
                          него похоже;
        hard_span_sec  -- ширина полосы трудных окон, считая от края зазора: с
                          обеих сторон приступа берутся окна, отстоящие от него
                          не дальше gap + hard_span_sec.

        Если трудных окон меньше запрошенного, недостача добирается обычным
        фоном -- размер выборки от настроек не зависит.

        Возвращает (x, y): x формы (окна, каналы, отсчёты), y -- метки 0/1.
        При return_groups=True возвращает (x, y, groups), где groups[i] --
        patient_key окна x[i], с учётом отбраковки и перемешивания. Это группы
        для patient_folds() / patient_cv(); для CV берите только fold="train".

        Ошибки чтения отдельных окон пропускаются с WARNING и traceback в
        логгере epi.windows, в том числе при verbose=False. Неполные окна
        тоже считаются ошибкой чтения, а не дополняются нулями.
        """
        generator = np.random.default_rng(seed)
        band = (self.gap_before_sec + hard_span_sec,
                self.gap_after_sec + hard_span_sec)

        def candidates(label, hard=None):
            where, params = "fold = ? AND label = ?", [fold, label]
            if hard is not None:
                where += " AND %s%s" % ("" if hard else "NOT ", self.NEAR)
                params += list(band)
            rows = self.conn.execute(
                """SELECT windows.*, recordings.patient_key, recordings.rel_path
                   FROM windows
                   JOIN recordings ON recordings.id = windows.recording_id
                   JOIN splits USING (patient_key)
                   WHERE %s""" % where, params,
            ).fetchall()
            return [rows[i] for i in generator.permutation(len(rows))]

        def fill(label, quota, hard=None):
            chosen, flat, unreadable = [], 0, 0

            for window in candidates(label, hard):
                if len(chosen) >= quota:
                    break

                try:
                    data = self.load(window)
                except (WindowReadError, IndexError, OSError, EOFError) as error:
                    unreadable += 1
                    logger.warning(
                        "Skipping unreadable window: fold=%s window_id=%s "
                        "recording_id=%s path=%r segment=%s start_sec=%s "
                        "length_sec=%s label=%s channels=%s target_rate=%s; %s: %s",
                        fold, window["id"], window["recording_id"], window["rel_path"],
                        window["segment"], window["start_sec"], window["length_sec"],
                        window["label"], self.channels, self.rate,
                        type(error).__name__, error, exc_info=True,
                    )
                    continue

                if self.flat_fraction(data) > max_flat:
                    flat += 1
                    continue

                chosen.append((data, window["patient_key"]))

            return chosen, flat, unreadable

        wanted_positive = int(round(n * positive_ratio))
        wanted_negative = n - wanted_positive
        wanted_hard = int(round(wanted_negative * hard_ratio))

        positives, dropped, unreadable = fill(1, wanted_positive)

        if wanted_hard:
            hard, dropped_hard, unreadable_hard = fill(
                0, wanted_hard, hard=True
            )
            easy, dropped_easy, unreadable_easy = fill(
                0, wanted_negative - len(hard), hard=False
            )
        else:
            hard, dropped_hard, unreadable_hard = [], 0, 0
            easy, dropped_easy, unreadable_easy = fill(
                0, wanted_negative
            )
        negatives = hard + easy

        kept = positives + negatives
        total_flat = dropped + dropped_hard + dropped_easy
        total_unreadable = unreadable + unreadable_hard + unreadable_easy
        if not kept:
            raise ValueError(
                "%s: не осталось ни одного окна; плоских %d, ошибок чтения %d"
                % (fold, total_flat, total_unreadable)
            )
        if len(kept) < n:
            logger.warning(
                "%s: requested %d windows (%d positive), kept %d (%d positive); "
                "flat=%d, unreadable=%d; candidate windows exhausted",
                fold, n, wanted_positive, len(kept), len(positives),
                total_flat, total_unreadable,
            )
        labels = [1] * len(positives) + [0] * len(negatives)

        order = generator.permutation(len(kept))
        x = np.stack([kept[i][0] for i in order])
        y = np.array([labels[i] for i in order])

        if verbose:
            print("%s: %d окон (%d %s), отброшено плоских %d ошибок чтения %d"
                  % (fold, len(y), int(y.sum()), self.positive_name,
                     total_flat, total_unreadable))
            if wanted_hard:
                print("   из фона трудных %d из %d: не дальше %.0f с до и %.0f с "
                      "после приступа" % (len(hard), len(negatives), band[0], band[1]))
            print("   форма %s при %g Гц" % (x.shape, self.rate))
        if return_groups:
            return x, y, np.array([kept[i][1] for i in order])
        return x, y

    def export(self, fold="train", out_dir="training", **kwargs):
        """Сохранить выборку фолда в файл .npz.

        Внутри лежат x, y, groups (patient_key каждого окна), channels и rate.
        Остальные аргументы такие же, как у sample(); groups сохраняются всегда.
        """
        kwargs["return_groups"] = True
        x, y, groups = self.sample(fold=fold, **kwargs)
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        path = out_dir / ("%s.npz" % fold)
        np.savez_compressed(path, x=x, y=y, groups=groups,
                            channels=np.array(self.channels), rate=self.rate)
        print("записано в %s" % path)
        return path
