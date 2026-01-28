import json
import platform
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import cv2


def detect_qr_type(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("{") or raw.startswith("["):
        return "json"
    if raw.lower().startswith(("http://", "https://")):
        return "url"
    if raw.upper().startswith("BEGIN:VCARD"):
        return "vcard"
    if raw.upper().startswith("WIFI:"):
        return "wifi"
    return "text"


def parse_qr_payload(text: str) -> dict:
    """
    Parse common QR payload formats:
      - JSON
      - URL
      - vCard
      - WiFi QR (WIFI:T:WPA;S:ssid;P:pass;;)
      - Fallback: raw text
    """
    raw = (text or "").strip()

    payload_type = detect_qr_type(raw)

    # JSON
    if payload_type == "json":
        try:
            return {"type": "json", "raw": raw, "data": json.loads(raw)}
        except json.JSONDecodeError:
            pass

    # URL
    if payload_type == "url":
        u = urlparse(raw)
        return {
            "type": "url",
            "raw": raw,
            "data": {
                "scheme": u.scheme,
                "netloc": u.netloc,
                "path": u.path,
                "params": u.params,
                "query": u.query,
                "query_params": {
                    k: v if len(v) != 1 else v[0]
                    for k, v in parse_qs(u.query).items()
                },
                "fragment": u.fragment,
            },
        }

    # vCard (very basic parsing)
    if payload_type == "vcard":
        fields = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.upper().startswith(("BEGIN:", "END:")):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip().upper()] = v.strip()
        return {"type": "vcard", "raw": raw, "data": fields}

    # WiFi QR format: WIFI:T:WPA;S:MySSID;P:MyPass;H:false;;
    if payload_type == "wifi":
        body = raw[5:]
        parts = [p for p in body.split(";") if p]
        data = {}
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                data[k.upper()] = _unescape_wifi_value(v)
        return {
            "type": "wifi",
            "raw": raw,
            "data": {
                "auth": data.get("T"),
                "ssid": data.get("S"),
                "password": data.get("P"),
                "hidden": (data.get("H", "").lower() == "true"),
            },
        }

    return {"type": payload_type, "raw": raw, "data": raw}


def _unescape_wifi_value(value: str) -> str:
    return (
        value.replace("\\\\", "\\")
        .replace("\\;", ";")
        .replace("\\:", ":")
        .replace("\\,", ",")
    )


def _to_gray(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _frame_brightness(frame) -> float:
    gray = _to_gray(frame)
    if gray is None:
        return 0.0
    return float(cv2.mean(gray)[0])


def _decode_qr_strings(frame, detector: cv2.QRCodeDetector) -> list[str]:
    results: list[str] = []
    gray = _to_gray(frame)
    if gray is None:
        return results

    try:
        ok, decoded_info, _points, _ = detector.detectAndDecodeMulti(gray)
        if ok and decoded_info:
            results = [s for s in decoded_info if s]
    except Exception:
        pass

    if not results:
        data, _points, _ = detector.detectAndDecode(gray)
        if data:
            results = [data]

    return results


def scan_qr_from_image(image_path: str) -> list[dict]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    det = cv2.QRCodeDetector()
    results = _decode_qr_strings(img, det)
    return [parse_qr_payload(s) for s in results]


def scan_qr_from_frame(frame) -> list[dict]:
    det = cv2.QRCodeDetector()
    results = _decode_qr_strings(frame, det)
    return [parse_qr_payload(s) for s in results]


def _candidate_backends() -> Iterable[int]:
    system = platform.system().lower()
    if system == "darwin":
        order = ["CAP_AVFOUNDATION", "CAP_ANY"]
    elif system == "windows":
        order = ["CAP_DSHOW", "CAP_MSMF", "CAP_ANY"]
    else:
        order = ["CAP_V4L2", "CAP_ANY"]

    backends = []
    for name in order:
        if hasattr(cv2, name):
            backends.append(getattr(cv2, name))
    if not backends:
        backends = [cv2.CAP_ANY]
    return backends


def _open_camera(camera_id: int, width: int, height: int) -> tuple[cv2.VideoCapture, int, float]:
    best = None
    for backend in _candidate_backends():
        cap = cv2.VideoCapture(camera_id, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        last_frame = None
        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None:
                last_frame = frame

        if last_frame is None:
            cap.release()
            continue

        brightness = _frame_brightness(last_frame)
        if best is None or brightness > best[2]:
            if best is not None:
                best[0].release()
            best = (cap, backend, brightness)
        else:
            cap.release()

    if best is None:
        raise RuntimeError(
            "Cannot open camera. If you're on macOS, allow camera access for Python/Terminal in System Settings."
        )

    return best


def scan_qr_from_camera(
    camera_id: int = 0,
    show_window: bool = True,
    width: int = 1280,
    height: int = 720,
    min_brightness: float = 5.0,
    parse_payload: bool = False,
):
    cap, backend, brightness = _open_camera(camera_id, width, height)
    if brightness < min_brightness:
        print(
            "Warning: camera frames are very dark. Check lens cover/permissions or try a different camera_id."
        )
    print(f"Camera opened (backend={backend}, brightness={brightness:.1f}). Press 'q' to quit.")

    det = cv2.QRCodeDetector()
    seen = set()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        decoded_strings = _decode_qr_strings(frame, det)

        for s in decoded_strings:
            if s not in seen:
                seen.add(s)
                print("\n=== QR Detected ===")
                print("Raw:", s)
                print("Type:", detect_qr_type(s))
                if parse_payload:
                    parsed = parse_qr_payload(s)
                    print("Parsed:", parsed["data"])

        if show_window:
            cv2.imshow("QR Scanner", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example 1: scan an image
    # print(scan_qr_from_image("qrcode.png"))

    # Example 2: scan from webcam
    scan_qr_from_camera(camera_id=0, show_window=True)
