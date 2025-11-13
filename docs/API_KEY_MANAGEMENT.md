# 🔑 API Key Management System

## Tổng Quan

Hệ thống API key tự động với các tính năng:
- ✅ **Tự động tạo key ngẫu nhiên** (32 ký tự hex)
- ✅ **Tự động hết hạn sau 90 ngày** (có thể tùy chỉnh)
- ✅ **Lưu trữ an toàn** (hash SHA-256)
- ✅ **Quản lý dễ dàng** (CLI tool)

---

## 📋 Cách Sử Dụng

### 1️⃣ Tạo API Key Mới

```bash
# Tạo key mặc định (90 ngày)
./scripts/manage_apikeys.sh create "Client ABC"

# Tạo key với thời hạn tùy chỉnh
./scripts/manage_apikeys.sh create "VIP Client" 180
```

**Output:**
```
✓ API key created for 'Client ABC'
  Expires: 2025-02-09
  Key: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
  ⚠️  Save this key! It won't be shown again.
```

⚠️ **Quan trọng:** Key chỉ hiển thị 1 lần duy nhất khi tạo!

---

### 2️⃣ Liệt Kê Tất Cả Keys

```bash
./scripts/manage_apikeys.sh list
```

**Output:**
```
================================================================================
API KEYS
================================================================================

✓ Active
  Name: Client ABC
  Hash: a1b2c3d4e5f6g7h8...
  Created: 2024-11-11
  Expires: 2025-02-09
  Days left: 90

✗ Inactive/Expired
  Name: Old Client
  Hash: x9y8z7w6v5u4t3s2...
  Created: 2024-08-01
  Expires: 2024-10-30
  Days left: -12
================================================================================
```

---

### 3️⃣ Gia Hạn API Key

```bash
# Gia hạn thêm 90 ngày
./scripts/manage_apikeys.sh renew a1b2c3d4e5f6g7h8

# Gia hạn thêm 180 ngày
./scripts/manage_apikeys.sh renew a1b2c3d4e5f6g7h8 180
```

---

### 4️⃣ Thu Hồi API Key

```bash
./scripts/manage_apikeys.sh revoke a1b2c3d4e5f6g7h8
```

---

### 5️⃣ Dọn Dẹp Keys Hết Hạn

```bash
./scripts/manage_apikeys.sh cleanup
```

---

## 🔐 Sử Dụng API Key

### cURL Example

```bash
curl -X POST \
  -H "X-API-Key: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6" \
  -F "file=@/path/to/image.png" \
  http://localhost:8000/api/v1/analyze
```

### Python Example

```python
import requests

API_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6"
headers = {"X-API-Key": API_KEY}

with open("image.png", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/v1/analyze",
        headers=headers,
        files=files
    )

print(response.json())
```

---

## 📁 Lưu Trữ

API keys được lưu trong:
```
/home/vbdi/Documents/convnext-chexpert-attention/full_stream/data/api_keys.json
```

**Format:**
```json
{
  "hash_of_key": {
    "name": "Client ABC",
    "created_at": "2024-11-11T10:30:00",
    "expires_at": "2025-02-09T10:30:00",
    "active": true,
    "metadata": {}
  }
}
```

⚠️ **Bảo mật:** Chỉ hash của key được lưu, không lưu key gốc!

---

## 🔄 Tự Động Hóa

### Cron Job - Dọn Dẹp Hàng Tuần

```bash
# Thêm vào crontab
crontab -e

# Chạy cleanup mỗi Chủ Nhật lúc 2:00 AM
0 2 * * 0 /path/to/scripts/manage_apikeys.sh cleanup
```

### Cron Job - Cảnh Báo Key Sắp Hết Hạn

```bash
# Script kiểm tra keys sắp hết hạn (< 7 ngày)
0 9 * * * /path/to/scripts/check_expiring_keys.sh
```

---

## 🆚 So Sánh: Legacy vs New System

| Feature | Legacy Keys | New System |
|---------|-------------|------------|
| **Tạo key** | Hard-coded | Auto-generated |
| **Bảo mật** | Plain text | SHA-256 hash |
| **Hết hạn** | ❌ Không | ✅ 90 ngày |
| **Quản lý** | Sửa code | CLI tool |
| **Gia hạn** | ❌ Không | ✅ Có |

---

## 🔧 Advanced Usage

### Sử Dụng Python Trực Tiếp

```python
from src.api.auth import create_api_key, list_api_keys, verify_api_key

# Tạo key
key = create_api_key("Client XYZ", expiration_days=90)
print(f"New key: {key}")

# Liệt kê keys
keys = list_api_keys()
for hash, info in keys.items():
    print(f"{info['name']}: {info['days_until_expiration']} days left")

# Verify key
info = verify_api_key(key)
if info:
    print(f"Valid key for: {info['name']}")
```

---

## ❓ FAQ

### Q: Key bị mất, làm sao lấy lại?
**A:** Không thể lấy lại! Phải tạo key mới và thu hồi key cũ.

### Q: Có thể thay đổi thời hạn mặc định?
**A:** Có, sửa `DEFAULT_EXPIRATION_DAYS` trong `src/api/auth.py`

### Q: Legacy keys vẫn hoạt động?
**A:** Có, để backward compatibility. Nhưng nên migrate sang hệ thống mới.

### Q: Làm sao biết key sắp hết hạn?
**A:** Dùng `./scripts/manage_apikeys.sh list` để xem "Days left"

---

## 🚀 Migration Guide

### Chuyển Từ Legacy Sang New System

```bash
# 1. Tạo keys mới cho tất cả clients
./scripts/manage_apikeys.sh create "Client 1" 90
./scripts/manage_apikeys.sh create "Client 2" 90

# 2. Gửi keys mới cho clients

# 3. Sau khi clients đã update, xóa legacy keys
# Sửa src/api/server.py:
LEGACY_API_KEYS = {}  # Xóa hết
```

---

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra file log: `data/api_keys.json`
2. Test với legacy key: `test_key_123`
3. Xem docs: `docs/QUICK_START.md`

