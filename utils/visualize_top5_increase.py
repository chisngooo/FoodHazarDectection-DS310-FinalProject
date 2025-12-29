import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu
print("Đang tải dữ liệu...")
df_original = pd.read_csv('data/train_data/original/train_original.csv')
df_augmented = pd.read_csv('data/train_data/aug1/aug_data1.csv')

# Tính toán thống kê tổng
increase = len(df_augmented) - len(df_original)
increase_percent = (increase / len(df_original)) * 100

# ============ Chuẩn bị dữ liệu Hazard Category (Top 5 tăng nhiều nhất) ============
hazard_categories = df_original['hazard-category'].value_counts()
hazard_cats = hazard_categories.index.tolist()
hazard_orig = [len(df_original[df_original['hazard-category'] == cat]) for cat in hazard_cats]
hazard_aug = [len(df_augmented[df_augmented['hazard-category'] == cat]) for cat in hazard_cats]
hazard_increases = [aug - orig for aug, orig in zip(hazard_aug, hazard_orig)]

# Sắp xếp theo số lượng tăng (giảm dần) và lấy top 5
sorted_hazard = sorted(zip(hazard_cats, hazard_orig, hazard_aug, hazard_increases), 
                      key=lambda x: x[3], reverse=True)[:5]
hazard_cats, hazard_orig, hazard_aug, hazard_increases = zip(*sorted_hazard)

# ============ Chuẩn bị dữ liệu Product Category (Top 5 tăng nhiều nhất) ============
product_categories = df_original['product-category'].value_counts()
product_cats = product_categories.index.tolist()
product_orig = [len(df_original[df_original['product-category'] == cat]) for cat in product_cats]
product_aug = [len(df_augmented[df_augmented['product-category'] == cat]) for cat in product_cats]
product_increases = [aug - orig for aug, orig in zip(product_aug, product_orig)]

# Sắp xếp theo số lượng tăng (giảm dần) và lấy top 5
sorted_product = sorted(zip(product_cats, product_orig, product_aug, product_increases), 
                       key=lambda x: x[3], reverse=True)[:5]
product_cats, product_orig, product_aug, product_increases = zip(*sorted_product)

width = 0.35

# ============ Tạo 1 figure với 2 subplots ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(f'Top 5 Categories with Highest Augmentation Increase', 
             fontsize=15, fontweight='bold', y=1.02)

# ============ CHART 1: Hazard Category (Top 5 tăng nhiều nhất) ============
x1 = np.arange(len(hazard_cats))
bars1_orig = ax1.bar(x1 - width/2, hazard_orig, width, 
                     label='Original', color='#3498db', edgecolor='black', linewidth=1, alpha=0.9)
bars1_aug = ax1.bar(x1 + width/2, hazard_aug, width, 
                    label='Augmented', color='#2ecc71', edgecolor='black', linewidth=1, alpha=0.9)

# Thêm số lượng trên cột
for bar in bars1_orig:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars1_aug:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Thêm số lượng tăng ở giữa
for i, inc in enumerate(hazard_increases):
    ax1.text(i, max(hazard_orig[i], hazard_aug[i]) * 0.5,
            f'+{inc}',
            ha='center', va='center', fontsize=10, fontweight='bold', 
            color='darkgreen', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax1.set_xlabel('Hazard Category', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Records', fontsize=11, fontweight='bold')
ax1.set_title('Hazard Category', fontsize=12, fontweight='bold', pad=10)
ax1.set_xticks(x1)
ax1.set_xticklabels(hazard_cats, rotation=45, ha='right', fontsize=10)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ============ CHART 2: Product Category (Top 5 tăng nhiều nhất) ============
x2 = np.arange(len(product_cats))
bars2_orig = ax2.bar(x2 - width/2, product_orig, width, 
                     label='Original', color='#3498db', edgecolor='black', linewidth=1, alpha=0.9)
bars2_aug = ax2.bar(x2 + width/2, product_aug, width, 
                    label='Augmented', color='#2ecc71', edgecolor='black', linewidth=1, alpha=0.9)

# Thêm số lượng trên cột
for bar in bars2_orig:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2_aug:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Thêm số lượng tăng ở giữa
for i, inc in enumerate(product_increases):
    ax2.text(i, max(product_orig[i], product_aug[i]) * 0.5,
            f'+{inc}',
            ha='center', va='center', fontsize=10, fontweight='bold', 
            color='darkgreen', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax2.set_xlabel('Product Category', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Records', fontsize=11, fontweight='bold')
ax2.set_title('Product Category', fontsize=12, fontweight='bold', pad=10)
ax2.set_xticks(x2)
ax2.set_xticklabels(product_cats, rotation=45, ha='right', fontsize=10)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('augmentation_top5_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Đã lưu: augmentation_top5_combined.png")

print(f"\n📈 Summary:")
print(f"   Total - Original: {len(df_original):,} → Augmented: {len(df_augmented):,} (+{increase_percent:.1f}%)")
print(f"\n🏆 Top 5 Hazard Categories:")
for i, (cat, orig, aug, inc) in enumerate(zip(hazard_cats, hazard_orig, hazard_aug, hazard_increases), 1):
    pct = (inc / orig * 100) if orig > 0 else 0
    print(f"   {i}. {cat}: {orig:,} → {aug:,} (+{inc:,}, +{pct:.1f}%)")

print(f"\n🏆 Top 5 Product Categories:")
for i, (cat, orig, aug, inc) in enumerate(zip(product_cats, product_orig, product_aug, product_increases), 1):
    pct = (inc / orig * 100) if orig > 0 else 0
    print(f"   {i}. {cat}: {orig:,} → {aug:,} (+{inc:,}, +{pct:.1f}%)")

plt.show()
