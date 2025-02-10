import json
import matplotlib.pyplot as plt
import seaborn as sns
with open("coordinates.json", 'r') as f:
    json_data = json.load(f)


heights = []
widths = []
areas = []
lengths=[]
for key, value in json_data.items():
    heights.append(value['height'])
    widths.append(value['width'])
    areas.append(value['area'])
    lengths.append(value['length'])


sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))


plt.subplot(1, 4, 1)
sns.histplot(heights, kde=True, color="skyblue", bins=10)
plt.title('Height Distribution')
plt.xlabel('Height')
plt.ylabel('Frequency')


plt.subplot(1, 4, 2)
sns.histplot(widths, kde=True, color="lightgreen", bins=10)
plt.title('Width Distribution')
plt.xlabel('Width')
plt.ylabel('Frequency')

plt.subplot(1, 4, 3)
sns.histplot(areas, kde=True, color="salmon", bins=10)
plt.title('Area Distribution')
plt.xlabel('Area')
plt.ylabel('Frequency')

plt.subplot(1, 4, 4)
sns.histplot(lengths, kde=True, color="salmon", bins=10)
plt.title('Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.tight_layout()

plt.show()
