"""ЛР №1, задание 11: три способа умножения матрицы на вектор."""

# Блок 1. Библиотеки и настройки
import os
# Настроить вычислительную библиотеку ДО импорта NumPy.
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

SIZES = (1000, 2000, 3000, 4000, 5000)
WORKERS = (2, 4)
REPEATS = 5


# Блок 2. Подключение процессов к общей матрице и вектору
def connect(matrix_memory, vector_memory, n):
    global A, x, shared_refs
    shared_refs = (matrix_memory, vector_memory)
    A = np.frombuffer(matrix_memory, dtype=np.float64).reshape(n, n)
    x = np.frombuffer(vector_memory, dtype=np.float64)
    A.flags.writeable = False
    x.flags.writeable = False


# Блок 3. Последовательное умножение
def sequential(matrix, vector):
    return matrix @ vector


# Блок 4. Работа одного процесса по строкам или столбцам
def rows_part(bounds):
    start, end = bounds
    return A[start:end, :] @ x


def columns_part(bounds):
    start, end = bounds
    return A[:, start:end] @ x[start:end]


# Блок 5. Разделение работы и сбор результата
def parallel(pool, n, p, mode):
    bounds = [(i * n // p, (i + 1) * n // p) for i in range(p)]
    if mode == "rows":
        parts = pool.map(rows_part, bounds)
        return np.concatenate(parts)
    parts = pool.map(columns_part, bounds)
    return np.sum(parts, axis=0)


# Блок 6. Прогрев, измерение и проверка результата
def measure(function, expected):
    np.testing.assert_allclose(function(), expected, rtol=1e-10, atol=1e-10)
    times = []
    error = 0.0
    for _ in range(REPEATS):
        start = perf_counter()
        result = function()
        times.append(perf_counter() - start)
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
        error = max(error, float(np.max(np.abs(result - expected))))
    return median(times), error


# Блок 7. Понятная строка сравнения и данные для CSV
def record(results, n, mode, p, seconds, baseline, error):
    speedup = baseline / seconds
    efficiency = speedup / p
    results.append([n, mode, p, seconds, speedup, efficiency, error])
    label = {"seq": "Последовательно", "rows": "По строкам",
             "columns": "По столбцам"}[mode]
    print(f"{n:5d}  {label:16s} {p:2d}  {seconds * 1000:9.3f}"
          f"  {speedup:7.3f}  {efficiency * 100:7.2f}", flush=True)


# Блок 8. Полное исследование при обычном запуске
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="только проверка на размерах 100 и 300")
    args = parser.parse_args()
    sizes = (100, 300) if args.quick else SIZES
    context = mp.get_context("spawn")
    results = []
    print("БЫСТРАЯ ПРОВЕРКА" if args.quick else "ПОЛНОЕ ИССЛЕДОВАНИЕ")
    print(f"ОС: {os.name}; NumPy: {np.__version__}; повторов: {REPEATS}")
    print("n      Способ            p    Время, мс        S      E, %")
    for n in sizes:
        matrix_memory = context.RawArray("d", n * n)
        vector_memory = context.RawArray("d", n)
        matrix = np.frombuffer(matrix_memory, dtype=np.float64).reshape(n, n)
        vector = np.frombuffer(vector_memory, dtype=np.float64)
        rng = np.random.default_rng(11 + n)
        rng.random(out=matrix)
        rng.random(out=vector)
        expected = sequential(matrix, vector)
        seq_time, error = measure(lambda: sequential(matrix, vector), expected)
        record(results, n, "seq", 1, seq_time, seq_time, error)
        for p in WORKERS:
            with context.Pool(p, initializer=connect,
                              initargs=(matrix_memory, vector_memory, n)) as pool:
                for mode in ("rows", "columns"):
                    seconds, error = measure(
                        lambda: parallel(pool, n, p, mode), expected)
                    record(results, n, mode, p, seconds, seq_time, error)

    # Блок 9. Сохранение сравнения рядом с программой
    filename = "quick_results.csv" if args.quick else "results.csv"
    path = Path(__file__).resolve().with_name(filename)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["n", "algorithm", "p", "seconds", "speedup",
                         "efficiency", "max_error"])
        writer.writerows(results)
    print("Все результаты проверены. S > 1: ускорение; S < 1: замедление.")
    print("E в CSV указана долей, в консоли - процентами.")
    print("Таблица:", path)
    if args.quick:
        print("Для отчёта запустите без --quick: нужны размеры 1000-5000.")


# Блок 10. Безопасный запуск дочерних процессов в Windows
if __name__ == "__main__":
    mp.freeze_support()
    main()
