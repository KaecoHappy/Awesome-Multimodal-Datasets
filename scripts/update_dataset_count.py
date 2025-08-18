import requests
from bs4 import BeautifulSoup

# 获取 README 文件内容
url = 'https://github.com/KaikoGit/Awesome-Multimodal-Datasets'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 提取所有链接
links = soup.find_all('a', href=True)

# 过滤出数据集链接
dataset_links = [link['href'] for link in links if 'dataset' in link['href'].lower()]

# 去重并统计数量
unique_datasets = set(dataset_links)
print(f"README 中提到的数据集数量：{len(unique_datasets)}")
