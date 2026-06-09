import ants

print("Testing ch2 loading...")
ch2 = ants.image_read(ants.get_ants_data('ch2'))
print("Shape:", ch2.shape)
print("Spacing:", ch2.spacing)

# Let's crop a 50x50x50 region in the middle
mid = [s//2 for s in ch2.shape]
lower = [m - 25 for m in mid]
upper = [m + 25 for m in mid]

crop = ants.crop_indices(ch2, lower, upper)
print("Crop shape:", crop.shape)

ants.plot(crop, filename='test_crop.png', axis=2)
print("Saved test_crop.png")
