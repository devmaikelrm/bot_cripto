<?php
/**
 * Binance Order Book snapshot fetcher (public REST).
 *
 * Saves snapshots as JSON Lines (one JSON object per line) for later analysis.
 *
 * Examples:
 *   php scripts/binance_orderbook.php --symbol BTCUSDT --limit 100 --out data/orderbook/BTCUSDT.jsonl
 *   php scripts/binance_orderbook.php --symbol ETHUSDT --limit 500 --out data/orderbook/ETHUSDT.jsonl --interval 1 --count 60
 *
 * Notes:
 * - This uses the public endpoint; no API key required.
 * - Symbols are in Binance format like BTCUSDT (no slash).
 */

declare(strict_types=1);

function stderr(string $msg): void {
    fwrite(STDERR, $msg . PHP_EOL);
}

function usage(int $rc = 1): int {
    $u = <<<TXT
Usage:
  php scripts/binance_orderbook.php --symbol BTCUSDT [--limit 5|10|20|50|100|500|1000|5000]
                                   [--out data/orderbook/BTCUSDT.jsonl]
                                   [--base https://api.binance.com]
                                   [--interval 1] [--count 1]
                                   [--timeout 10]

Saves JSONL with fields:
  ts_unix_ms, ts_iso_utc, symbol, limit, lastUpdateId, bids, asks

TXT;
    stderr($u);
    return $rc;
}

/**
 * Minimal CLI arg parsing: --key value (no short flags).
 * @return array<string, string>
 */
function parseArgs(array $argv): array {
    $out = [];
    $n = count($argv);
    for ($i = 1; $i < $n; $i++) {
        $a = $argv[$i];
        if (!str_starts_with($a, "--")) {
            continue;
        }
        $key = substr($a, 2);
        $val = "1";
        if ($i + 1 < $n && !str_starts_with($argv[$i + 1], "--")) {
            $val = $argv[$i + 1];
            $i++;
        }
        $out[$key] = $val;
    }
    return $out;
}

function ensureDir(string $path): void {
    $dir = dirname($path);
    if ($dir === "." || $dir === "") {
        return;
    }
    if (!is_dir($dir)) {
        if (!mkdir($dir, 0775, true) && !is_dir($dir)) {
            throw new RuntimeException("Failed to create dir: " . $dir);
        }
    }
}

/**
 * @return array<string, mixed>
 */
function httpGetJson(string $url, int $timeoutS): array {
    $ch = curl_init();
    if ($ch === false) {
        throw new RuntimeException("curl_init failed");
    }

    curl_setopt_array($ch, [
        CURLOPT_URL => $url,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_CONNECTTIMEOUT => $timeoutS,
        CURLOPT_TIMEOUT => $timeoutS,
        CURLOPT_HTTPHEADER => [
            "Accept: application/json",
            "User-Agent: bot-cripto/binance_orderbook.php",
        ],
    ]);

    $body = curl_exec($ch);
    $errno = curl_errno($ch);
    $err = curl_error($ch);
    $httpCode = (int)curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($body === false || $errno !== 0) {
        throw new RuntimeException("cURL error ($errno): $err");
    }
    if ($httpCode < 200 || $httpCode >= 300) {
        throw new RuntimeException("HTTP $httpCode: " . substr((string)$body, 0, 500));
    }

    $json = json_decode((string)$body, true);
    if (!is_array($json)) {
        throw new RuntimeException("Invalid JSON response");
    }
    return $json;
}

function nowUnixMs(): int {
    return (int)floor(microtime(true) * 1000);
}

function nowIsoUtc(): string {
    $dt = new DateTimeImmutable("now", new DateTimeZone("UTC"));
    return $dt->format("Y-m-d\\TH:i:s.v\\Z");
}

function main(array $argv): int {
    if (!function_exists("curl_init")) {
        stderr("Missing PHP cURL extension (curl).");
        return 2;
    }

    $args = parseArgs($argv);
    $symbol = strtoupper(trim($args["symbol"] ?? ""));
    if ($symbol === "") {
        stderr("Missing --symbol (e.g. BTCUSDT).");
        return usage(2);
    }

    $limit = (int)($args["limit"] ?? "100");
    $allowedLimits = [5, 10, 20, 50, 100, 500, 1000, 5000];
    if (!in_array($limit, $allowedLimits, true)) {
        stderr("Invalid --limit. Allowed: " . implode(", ", $allowedLimits));
        return usage(2);
    }

    $base = rtrim((string)($args["base"] ?? "https://api.binance.com"), "/");
    $outPath = (string)($args["out"] ?? ("data/orderbook/" . $symbol . ".jsonl"));
    $intervalS = (int)($args["interval"] ?? "0");
    $count = (int)($args["count"] ?? "1");
    $timeoutS = (int)($args["timeout"] ?? "10");

    if ($count < 1) {
        stderr("--count must be >= 1");
        return usage(2);
    }
    if ($intervalS < 0) {
        stderr("--interval must be >= 0");
        return usage(2);
    }
    if ($timeoutS < 1 || $timeoutS > 120) {
        stderr("--timeout must be between 1 and 120 seconds");
        return usage(2);
    }

    ensureDir($outPath);

    $query = http_build_query([
        "symbol" => $symbol,
        "limit" => $limit,
    ]);
    $url = $base . "/api/v3/depth?" . $query;

    for ($i = 0; $i < $count; $i++) {
        $tsMs = nowUnixMs();
        $tsIso = nowIsoUtc();

        try {
            $data = httpGetJson($url, $timeoutS);
        } catch (Throwable $e) {
            stderr("Fetch failed: " . $e->getMessage());
            // Best-effort wait before next attempt (if any).
            if ($i + 1 < $count && $intervalS > 0) {
                sleep($intervalS);
            }
            continue;
        }

        $line = [
            "ts_unix_ms" => $tsMs,
            "ts_iso_utc" => $tsIso,
            "symbol" => $symbol,
            "limit" => $limit,
            "lastUpdateId" => $data["lastUpdateId"] ?? null,
            "bids" => $data["bids"] ?? null,
            "asks" => $data["asks"] ?? null,
        ];

        $jsonLine = json_encode($line, JSON_UNESCAPED_SLASHES);
        if ($jsonLine === false) {
            stderr("json_encode failed");
            return 3;
        }

        $ok = file_put_contents($outPath, $jsonLine . PHP_EOL, FILE_APPEND | LOCK_EX);
        if ($ok === false) {
            stderr("Failed to write: " . $outPath);
            return 3;
        }

        // Small status to stdout (so it's easy to see it's working).
        fwrite(STDOUT, "OK $symbol limit=$limit -> $outPath ($tsIso)\n");

        if ($i + 1 < $count && $intervalS > 0) {
            sleep($intervalS);
        }
    }

    return 0;
}

exit(main($argv));
