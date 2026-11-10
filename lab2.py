"""ЛР №2, вариант 5. Python 3.9+, только стандартная библиотека."""

# Задаю число буферов, их размер и задержки для модели датчиков.
import argparse
import csv
from pathlib import Path
import random
from statistics import median
from threading import BoundedSemaphore, Condition, Lock, Thread
from time import perf_counter, sleep

N_BUFFERS, SIZE = 5, 5
FREE, WRITING, FULL, PROCESSING = range(4)
T_GAP, T_ELEMENT, T_PROCESS = 0.010, 0.002, 0.015


# Подготавливаю буферы, состояния и блокировки для одного опыта.
def simulate(n=3, m=1, packets=60, seed=5):
    if n < 1 or m < 1 or packets < 2 or packets % 2:
        raise ValueError("n и m >= 1; packets >= 2 и чётное")
    buffers = [[0] * SIZE for _ in range(N_BUFFERS)]
    states = [FREE] * N_BUFFERS
    packet_ids = [None] * N_BUFFERS
    free_slots = BoundedSemaphore(N_BUFFERS)
    changed = Condition()
    result_lock = Lock()
    buf4, results, processed, expected = [], [], [], {}
    waits, held_times, full_times = [], [], []
    reserved_at, full_at = [0.0] * N_BUFFERS, [0.0] * N_BUFFERS
    sensors_left = n

    # Датчик ждёт разрешения семафора и занимает свободный буфер.
    def sensor(sid):
        nonlocal sensors_left
        rng = random.Random(seed + sid)
        for packet_id in range(sid, packets, n):
            sleep(T_GAP)
            start_wait = perf_counter()
            free_slots.acquire()  # Если все буферы заняты, датчик ждёт здесь.
            waited = perf_counter() - start_wait
            with changed:
                i = states.index(FREE)
                states[i] = WRITING
                reserved_at[i] = perf_counter()
                packet_ids[i] = packet_id
                waits.append(waited)

            # После записи всех пяти чисел разрешаю обработчику взять буфер.
            data = [rng.randint(-100, 100) for _ in range(SIZE)]
            for k, value in enumerate(data):
                sleep(T_ELEMENT)
                buffers[i][k] = value
            with changed:
                expected[packet_id] = (min(data), max(data))
                states[i] = FULL
                full_at[i] = perf_counter()
                changed.notify()
        with changed:
            sensors_left -= 1
            changed.notify_all()

    # Обработчик ищет готовые данные, начиная со следующего буфера.
    def worker(wid):
        pos = wid % N_BUFFERS
        while True:
            with changed:
                while True:
                    i = None
                    for k in range(N_BUFFERS):
                        candidate = (pos + k) % N_BUFFERS
                        if states[candidate] == FULL:
                            i = candidate
                            break
                    if i is not None:
                        states[i] = PROCESSING
                        full_times.append(perf_counter() - full_at[i])
                        break
                    if sensors_left == 0:
                        return
                    changed.wait()

            # Нахожу минимум и максимум; каждые четыре числа складываю.
            sleep(T_PROCESS)
            pair = (min(buffers[i]), max(buffers[i]))
            with result_lock:
                processed.append((packet_ids[i], pair))
                buf4.extend(pair)
                if len(buf4) == 4:
                    results.append(sum(buf4))
                    buf4.clear()

            # Освобождаю входной буфер после передачи пары в общий результат.
            with changed:
                held_times.append(perf_counter() - reserved_at[i])
                states[i] = FREE
            free_slots.release()  # Теперь следующий датчик может занять буфер.
            pos = (i + 1) % N_BUFFERS

    # Запускаю датчики и обработчики, затем жду окончания всех потоков.
    workers = [Thread(target=worker, args=(w,)) for w in range(m)]
    sensors = [Thread(target=sensor, args=(s,)) for s in range(n)]
    start = perf_counter()
    for thread in workers + sensors:
        thread.start()
    for thread in sensors + workers:
        thread.join()
    elapsed = perf_counter() - start

    # Проверяю, что каждый пакет обработан один раз, и считаю показатели.
    ids = [pid for pid, pair in processed]
    if sorted(ids) != list(range(packets)):
        raise RuntimeError("Пакет потерян или обработан повторно")
    if any(pair != expected[pid] for pid, pair in processed):
        raise RuntimeError("Неверные min/max")
    if states != [FREE] * N_BUFFERS or buf4:
        raise RuntimeError("Остались незавершённые данные")
    if sum(results) != sum(sum(pair) for pair in expected.values()):
        raise RuntimeError("Неверная сумма результатов")
    return dict(n=n, m=m, seconds=elapsed, rate=packets / elapsed,
                wait_ms=1000 * sum(waits) / packets,
                occupied=100 * sum(held_times) / (N_BUFFERS * elapsed),
                full=100 * sum(full_times) / (N_BUFFERS * elapsed),
                results=results)


# Сравниваю девять сочетаний числа датчиков и обработчиков.
def benchmark(packets):
    rows = []
    metrics = ("seconds", "rate", "wait_ms", "occupied", "full")
    for n in (1, 3, 8):
        for m in (1, 2, 4):
            runs = [simulate(n, m, packets, seed=100 + r)
                    for r in range(3)]
            row = dict(n=n, m=m)
            row.update({key: median(run[key] for run in runs)
                        for key in metrics})
            rows.append(row)
            print(f"n={n}, m={m}: {row['rate']:.1f} пак/с; "
                  f"ожидание {row['wait_ms']:.2f} мс; "
                  f"занято {row['occupied']:.1f}%; FULL {row['full']:.1f}%",
                  flush=True)
    path = Path(__file__).resolve().with_name("experiment.csv")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Полная таблица:", path)


# Читаю параметры запуска и выбираю один опыт или исследование.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=3, help="число датчиков")
    parser.add_argument("-m", type=int, default=1, help="число обработчиков")
    parser.add_argument("-k", type=int, default=60, help="чётное число пакетов")
    parser.add_argument("--bench", action="store_true", help="исследование")
    parser.add_argument("--single", action="store_true", help="только один опыт")
    args = parser.parse_args()
    if args.bench and args.single:
        parser.error("Выберите только один режим: --bench или --single")
    if args.n < 1 or args.m < 1 or args.k < 2 or args.k % 2:
        parser.error("n и m >= 1; k >= 2 и чётное")
    if args.bench:
        benchmark(args.k)
    else:
        r = simulate(args.n, args.m, args.k)
        print("Суммы Буф4:", r["results"])
        print(f"Время: {r['seconds']:.3f} с; {r['rate']:.1f} пакетов/с")
        print(f"Ожидание датчика: {r['wait_ms']:.2f} мс")
        print(f"Занятость: {r['occupied']:.1f}%; FULL: {r['full']:.1f}%")
        print("Каждый пакет обработан ровно один раз. Проверки пройдены.")
        if not args.single:
            print("\nСравнение числа датчиков и обработчиков:")
            benchmark(args.k)


# Запускаю программу, только если файл открыт напрямую.
if __name__ == "__main__":
    main()
