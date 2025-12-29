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

# ============ Chuẩn bị dữ liệu Hazard Category (Toàn bộ) ============
hazard_categories = df_original['hazard-category'].value_counts()
hazard_cats = hazard_categories.index.tolist()
hazard_orig = [len(df_original[df_original['hazard-category'] == cat]) for cat in hazard_cats]
hazard_aug = [len(df_augmented[df_augmented['hazard-category'] == cat]) for cat in hazard_cats]

# Sắp xếp theo số lượng original (giảm dần) - Hiển thị toàn bộ
sorted_hazard = sorted(zip(hazard_cats, hazard_orig, hazard_aug), key=lambda x: x[1], reverse=True)
hazard_cats, hazard_orig, hazard_aug = zip(*sorted_hazard)

# ============ Chuẩn bị dữ liệu Product Category (Top 10 tăng nhiều nhất) ============
product_categories = df_original['product-category'].value_counts()
product_cats = product_categories.index.tolist()
product_orig = [len(df_original[df_original['product-category'] == cat]) for cat in product_cats]
product_aug = [len(df_augmented[df_augmented['product-category'] == cat]) for cat in product_cats]
product_increases = [aug - orig for aug, orig in zip(product_aug, product_orig)]

# Sắp xếp theo số lượng tăng (giảm dần) và lấy top 10
sorted_product = sorted(zip(product_cats, product_orig, product_aug, product_increases), 
                       key=lambda x: x[3], reverse=True)[:10]
product_cats, product_orig, product_aug, product_increases = zip(*sorted_product)

width = 0.35

# ============ CHART 1: Hazard Category ============
fig1, ax1 = plt.subplots(figsize=(16, 6))

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

ax1.set_xlabel('Hazard Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
ax1.set_title(f'Data Augmentation: All Hazard Categories', 
              fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x1)
ax1.set_xticklabels(hazard_cats, rotation=45, ha='right', fontsize=10)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('augmentation_hazard_category.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Đã lưu: augmentation_hazard_category.png")
plt.close()

# ============ CHART 2: Product Category (Top 10) ============
fig2, ax2 = plt.subplots(figsize=(16, 6))

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

ax2.set_xlabel('Product Category', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
ax2.set_title(f'Data Augmentation: Top 10 Product Categories with Highest Increase', 
              fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x2)
ax2.set_xticklabels(product_cats, rotation=45, ha='right', fontsize=10)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('augmentation_product_category.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Đã lưu: augmentation_product_category.png")

print(f"\n📈 Summary:")
print(f"   Original: {len(df_original):,} → Augmented: {len(df_augmented):,} (+{increase_percent:.1f}%)")
plt.show()
