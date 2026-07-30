"""Decimated analysis and synthesis filter banks over row-major buffers."""

from std.algorithm import parallelize
from std.sys import simd_width_of
from std.sys.info import num_physical_cores

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime ANALYSIS_PARALLEL_WORK = 4_000_000
comptime SYNTHESIS_PARALLEL_WORK = 16_000_000


def positive_mod(value: Int, modulus: Int) -> Int:
    var r = value % modulus
    if r < 0:
        r += modulus
    return r


def extended_sample(x: Ptr, n: Int, index: Int, mode: Int) -> Float64:
    if index >= 0 and index < n:
        return x[index]
    if mode == 0:
        return 0.0
    if mode == 1:
        return x[0] if index < 0 else x[n - 1]
    if mode == 3:
        return x[positive_mod(index, n)]
    if mode == 4:
        if n == 1:
            return x[0]
        if index < 0:
            return x[0] + Float64(index) * (x[1] - x[0])
        return x[n - 1] + Float64(index - n + 1) * (x[n - 1] - x[n - 2])
    if mode == 2 or mode == 6:
        var r = positive_mod(index, n)
        var q = (index - r) // n
        var mapped = r if positive_mod(q, 2) == 0 else n - 1 - r
        var value = x[mapped]
        if mode == 6 and positive_mod(q, 2) != 0:
            value = -value
        return value
    if n == 1:
        return x[0]
    var span = n - 1
    var r = positive_mod(index, span)
    var q = (index - r) // span
    var even = positive_mod(q, 2) == 0
    var mapped = r if even else n - 1 - r
    if mode == 5:
        return x[mapped]
    if even:
        return x[mapped] + Float64(q) * (x[n - 1] - x[0])
    return (
        -x[mapped]
        + Float64(q + 1) * x[n - 1]
        - Float64(q - 1) * x[0]
    )


def periodized_sample(x: Ptr, n: Int, index: Int) -> Float64:
    var even_n = n if n % 2 == 0 else n + 1
    var mapped = positive_mod(index, even_n)
    if mapped == n:
        mapped = n - 1
    return x[mapped]


def dwt_rows(
    x: Ptr,
    rows: Int,
    n: Int,
    dec_lo: Ptr,
    dec_hi: Ptr,
    filter_len: Int,
    approx: Ptr,
    detail: Ptr,
    coeff_len: Int,
    mode: Int,
):
    @parameter
    def transform_position(position: Int):
        var row = 0
        var k = position
        if rows != 1:
            row = position // coeff_len
            k = position - row * coeff_len
        var src = x + row * n
        var dst_a = approx + row * coeff_len
        var dst_d = detail + row * coeff_len
        var a = Float64(0.0)
        var d = Float64(0.0)
        var center = 2 * k + 1
        if mode == 8:
            center = 2 * k + filter_len // 2
        var start = center - filter_len + 1
        if start >= 0 and center < n:
            comptime W = simd_width_of[DType.float64]()
            var a_vec = SIMD[DType.float64, W](0.0)
            var d_vec = SIMD[DType.float64, W](0.0)
            var f = 0
            while f + W <= filter_len:
                var values = src.load[width=W](start + f)
                a_vec += dec_lo.load[width=W](f) * values
                d_vec += dec_hi.load[width=W](f) * values
                f += W
            a += a_vec.reduce_add()
            d += d_vec.reduce_add()
            while f < filter_len:
                var value = src[start + f]
                a += dec_lo[f] * value
                d += dec_hi[f] * value
                f += 1
        else:
            for f in range(filter_len):
                var value = (
                    periodized_sample(src, n, start + f)
                    if mode == 8
                    else extended_sample(src, n, start + f, mode)
                )
                a += dec_lo[f] * value
                d += dec_hi[f] * value
        dst_a[k] = a
        dst_d[k] = d

    var total = rows * coeff_len
    var work = total * filter_len

    if work >= ANALYSIS_PARALLEL_WORK:
        var workers = min(num_physical_cores(), 16)
        var tasks = workers * 4

        @parameter
        def process(task: Int):
            var begin = total * task // tasks
            var end = total * (task + 1) // tasks
            for position in range(begin, end):
                transform_position(position)

        parallelize[process](tasks, workers)
    else:
        for position in range(total):
            transform_position(position)


def idwt_rows(
    approx: Ptr,
    detail: Ptr,
    rows: Int,
    coeff_len: Int,
    rec_lo: Ptr,
    rec_hi: Ptr,
    filter_len: Int,
    dst: Ptr,
    result_len: Int,
    periodization: Bool,
):
    @parameter
    def reconstruct_pair(position: Int):
        var pair_len = result_len // 2
        var row = 0
        var q = position
        if rows != 1:
            row = position // pair_len
            q = position - row * pair_len
        var a = approx + row * coeff_len + q
        var d = detail + row * coeff_len + q
        var result = dst + row * result_len
        var half = filter_len // 2
        var even = Float64(0.0)
        var odd = Float64(0.0)
        comptime W = simd_width_of[DType.float64]()
        var j = 0
        while j + W <= half:
            var av = a.load[width=W](j)
            var dv = d.load[width=W](j)
            even += (
                av * rec_lo.load[width=W](j)
                + dv * rec_hi.load[width=W](j)
            ).reduce_add()
            odd += (
                av * rec_lo.load[width=W](half + j)
                + dv * rec_hi.load[width=W](half + j)
            ).reduce_add()
            j += W
        while j < half:
            even += a[j] * rec_lo[j] + d[j] * rec_hi[j]
            odd += (
                a[j] * rec_lo[half + j]
                + d[j] * rec_hi[half + j]
            )
            j += 1
        result[2 * q] = even
        result[2 * q + 1] = odd

    @parameter
    def reconstruct_position(position: Int):
        var row = position // result_len
        var i = position - row * result_len
        var a = approx + row * coeff_len
        var d = detail + row * coeff_len
        var result = dst + row * result_len
        var value = Float64(0.0)
        for f in range(filter_len):
            var numerator = i - f + filter_len - 2
            if periodization:
                numerator = positive_mod(
                    i + filter_len // 2 - 1 - f, result_len
                )
            if numerator % 2 == 0:
                var k = numerator // 2
                if k >= 0 and k < coeff_len:
                    value += a[k] * rec_lo[f] + d[k] * rec_hi[f]
        result[i] = value

    var total = rows * result_len
    if not periodization and filter_len % 2 == 0:
        var pairs = total // 2
        if total * filter_len >= SYNTHESIS_PARALLEL_WORK:
            var workers = min(num_physical_cores(), 16)
            var tasks = workers * 4

            @parameter
            def process_pairs(task: Int):
                var begin = pairs * task // tasks
                var end = pairs * (task + 1) // tasks
                for position in range(begin, end):
                    reconstruct_pair(position)

            parallelize[process_pairs](tasks, workers)
        else:
            for position in range(pairs):
                reconstruct_pair(position)
    elif rows > 1 and total * filter_len >= SYNTHESIS_PARALLEL_WORK:
        var workers = min(num_physical_cores(), 16)
        var tasks = workers * 4

        @parameter
        def process(task: Int):
            var begin = total * task // tasks
            var end = total * (task + 1) // tasks
            for position in range(begin, end):
                reconstruct_position(position)

        parallelize[process](tasks, workers)
    else:
        for row in range(rows):
            var a = approx + row * coeff_len
            var d = detail + row * coeff_len
            var result = dst + row * result_len
            for i in range(result_len):
                result[i] = 0.0
            for k in range(coeff_len):
                var base = 2 * k - filter_len + 2
                if periodization:
                    base = 2 * k - (filter_len // 2 - 1)
                for f in range(filter_len):
                    var i = base + f
                    if periodization:
                        i = positive_mod(i, result_len)
                        result[i] += a[k] * rec_lo[f] + d[k] * rec_hi[f]
                    elif i >= 0 and i < result_len:
                        result[i] += a[k] * rec_lo[f] + d[k] * rec_hi[f]


@export("mpw_dwt_f64")
def mpw_dwt_f64(
    x: Int,
    rows: Int,
    n: Int,
    dec_lo: Int,
    dec_hi: Int,
    filter_len: Int,
    approx: Int,
    detail: Int,
    coeff_len: Int,
    mode: Int,
) abi("C") -> Int:
    if (
        x == 0 or dec_lo == 0 or dec_hi == 0 or approx == 0 or detail == 0
        or rows < 0 or n < 1 or filter_len < 2 or coeff_len < 1
        or mode < 0 or mode > 8
    ):
        return 1
    var expected = (
        (n + 1) // 2
        if mode == 8
        else (n + filter_len - 1) // 2
    )
    if coeff_len != expected:
        return 2
    dwt_rows(
        Ptr(unsafe_from_address=x),
        rows,
        n,
        Ptr(unsafe_from_address=dec_lo),
        Ptr(unsafe_from_address=dec_hi),
        filter_len,
        Ptr(unsafe_from_address=approx),
        Ptr(unsafe_from_address=detail),
        coeff_len,
        mode,
    )
    return 0


@export("mpw_idwt_f64")
def mpw_idwt_f64(
    approx: Int,
    detail: Int,
    rows: Int,
    coeff_len: Int,
    rec_lo: Int,
    rec_hi: Int,
    filter_len: Int,
    dst: Int,
    result_len: Int,
    periodization: Int,
) abi("C") -> Int:
    if (
        approx == 0 or detail == 0 or rec_lo == 0 or rec_hi == 0 or dst == 0
        or rows < 0 or coeff_len < 1 or filter_len < 2 or result_len < 1
        or (periodization != 0 and periodization != 1)
    ):
        return 1
    var expected = (
        2 * coeff_len
        if periodization != 0
        else 2 * coeff_len - filter_len + 2
    )
    if result_len != expected:
        return 2
    idwt_rows(
        Ptr(unsafe_from_address=approx),
        Ptr(unsafe_from_address=detail),
        rows,
        coeff_len,
        Ptr(unsafe_from_address=rec_lo),
        Ptr(unsafe_from_address=rec_hi),
        filter_len,
        Ptr(unsafe_from_address=dst),
        result_len,
        periodization != 0,
    )
    return 0
