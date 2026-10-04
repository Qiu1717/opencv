import base64
import io
import numpy as np
import cv2

from config import get_logger

logger = get_logger(__name__)


def decode_image(source):
    if isinstance(source, str):
        if source.startswith('data:image'):
            source = source.split(',', 1)[1]
        data = base64.b64decode(source)
    elif isinstance(source, bytes):
        data = source
    else:
        logger.error(f"decode_image: unsupported type {type(source)}")
        return None

    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame is None:
        logger.error(f"decode_image: cv2.imdecode failed, {len(data)} bytes")

    return frame


def decode_upload(request):
    if 'image' in request.files:
        file = request.files['image']
        buf = io.BytesIO()
        file.save(buf)
        return decode_image(buf.getvalue())
    if 'image_base64' in request.form:
        return decode_image(request.form.get('image_base64'))
    return None