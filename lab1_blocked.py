"""ЛР 1, вариант 11. Блочное умножение без хранения всей матрицы.

Полный запуск: python lab1_blocked.py
Проверка: python lab1_blocked.py --quick
Свои размеры: python lab1_blocked.py --sizes 10000,100000 --repeats 1
Время включает генерацию блоков. Большие размеры не пропускаются.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import csv
import multiprocessing as mp
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np

SIZES = (1000, 2000, 3000, 4000, 5000, 10000, 100000, 1000000)


# Каждый процесс хранит только исходный вектор и настройки генератора.
def setup(n, block_size, seed):
    global N, BLOCK, SEED, X
    N, BLOCK, SEED = n, block_size, seed
    X = np.random.default_rng([seed, n]).random(n)


# Координаты блока определяют его числа независимо от порядка вычислений.
# При одинаковых n, seed и block_size все способы используют одну матрицу.
def matrix_block(row, col):
    height = min(BLOCK, N - row * BLOCK)
    width = min(BLOCK, N - col * BLOCK)
    rng = np.random.default_rng([SEED, N, row, col])
    return rng.random((height, width))


# Одно задание вычисляет несколько горизонтальных полос матрицы.
def rows_task(bounds):
    first, last = bounds
    begin = first * BLOCK
    answer = np.zeros(min(last * BLOCK, N) - begin)
    count = (N + BLOCK - 1) // BLOCK
    for row in range(first, last):
        offset = row * BLOCK - begin
        height = min(BLOCK, N - row * BLOCK)
        for col in range(count):
            tile = matrix_block(row, col)
            start = col * BLOCK
            answer[offset:offset + height] += tile @ X[start:start + tile.shape[1]]
            del tile
    return begin, answer


# Вертикальные полосы дают частичный вклад во весь вектор ответа.
def columns_task(bounds):
    first, last = bounds
    answer = np.zeros(N)
    count = (N + BLOCK - 1) // BLOCK
    for col in range(first, last):
        start = col * BLOCK
        for row in range(count):
            tile = matrix_block(row, col)
            offset = row * BLOCK
            answer[offset:offset + tile.shape[0]] += tile @ X[start:start + tile.shape[1]]
            del tile
    return answer


def parallel(pool, workers, mode):
    count = (N + BLOCK - 1) // BLOCK
    bounds = [(i * count // workers, (i + 1) * count // workers)
              for i in range(workers)]
    result = np.zeros(N)
    if mode == "rows":
        for start, values in pool.imap_unordered(rows_task, bounds):
            result[start:start + len(values)] = values
    else:
        for values in pool.imap(columns_task, bounds):
            result += values
    return result


# Полный прогрев не делаю: для огромных матриц это ещё один дорогой проход.
# Проверка ответа выполняется после остановки таймера.
def measure(function, repeats, expected=None):
    times = []
    error = 0.0
    reference = expected
    for repeat in range(repeats):
        print(f"    Повтор {repeat + 1}/{repeats}", flush=True)
        start = perf_counter()
        result = function()
        times.append(perf_counter() - start)
        if reference is None:
            reference = result
        else:
            np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)
            error = max(error, float(np.max(np.abs(result - reference))))
    return median(times), reference, error


def positive_list(text):
    try:
        values = [int(x) for x in text.split(",")]
        if not values or min(values) < 1:
            raise ValueError
        return list(dict.fromkeys(values))
    except ValueError:
        raise argparse.ArgumentTypeError("Нужны положительные целые через запятую")


# Маленькую матрицу собираю полностью только для независимой проверки алгоритмов.
def check(context):
    setup(17, 6, 11)
    dense = np.block([[matrix_block(r, c) for c in range(3)] for r in range(3)])
    expected = dense @ X
    np.testing.assert_allclose(rows_task((0, 3))[1], expected)
    with context.Pool(4, initializer=setup, initargs=(17, 6, 11)) as pool:
        for mode in ("rows", "columns"):
            np.testing.assert_allclose(parallel(pool, 4, mode), expected)
    print("Проверка на матрице 17x17 пройдена.", flush=True)


def ready():
    # Ответ подтверждает, что рабочий процесс готов принимать задания.
    return os.getpid()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=positive_list, default=SIZES)
    parser.add_argument("--workers", type=positive_list, default=(2, 4))
    parser.add_argument("--block", type=int, default=512, help="сторона блока")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.block < 1 or args.repeats < 1 or args.seed < 0:
        parser.error("block и repeats должны быть > 0, seed >= 0")
    sizes = (100, 300) if args.quick else args.sizes
    context = mp.get_context("spawn")
    check(context)
    name = "quick_blocked_results.csv" if args.quick else "blocked_results.csv"
    path = Path(__file__).resolve().with_name(name)
    print("Матрица генерируется блоками; генерация входит в замер.")
    print("Все размеры выполняются. 100000 и 1000000 могут считать очень долго.")
    print("Один блок занимает примерно", args.block**2 * 8 / 1024**2, "МиБ на процесс.")
    print("Дополнительно память нужна векторам, результатам и процессам.")
    print("Результаты:", path, flush=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["n", "algorithm", "workers", "seconds", "speedup",
                         "efficiency", "max_error", "block_size", "repeats", "seed"])

        def save(n, mode, workers, seconds, baseline, error):
            speedup = baseline / seconds
            writer.writerow([n, mode, workers, seconds, speedup, speedup / workers,
                             error, args.block, args.repeats, args.seed])
            file.flush()
            print(f"n={n}, {mode}, p={workers}: {seconds:.6f} с; "
                  f"S={speedup:.3f}; E={100 * speedup / workers:.2f}%", flush=True)

        for n in sizes:
            setup(n, args.block, args.seed)
            count = (n + args.block - 1) // args.block
            print(f"\nМатрица {n}x{n}: {count**2} блоков за один проход.", flush=True)
            print("Последовательное вычисление", flush=True)
            seq_time, expected, error = measure(
                lambda: rows_task((0, count))[1], args.repeats)
            save(n, "seq", 1, seq_time, seq_time, error)
            for workers in args.workers:
                with context.Pool(workers, initializer=setup,
                                  initargs=(n, args.block, args.seed)) as pool:
                    # Лёгкое задание не является полным прогревом каждого рабочего.
                    pool.apply(ready)
                    for mode in ("rows", "columns"):
                        print(f"Способ {mode}, процессов {workers}", flush=True)
                        seconds, _, error = measure(
                            lambda: parallel(pool, workers, mode), args.repeats, expected)
                        save(n, mode, workers, seconds, seq_time, error)
            del expected
    print("Все запрошенные размеры завершены, ответы проверены.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
