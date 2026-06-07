import pystac_client
import planetary_computer

catalog = pystac_client.Client.open(
    'https://planetarycomputer.microsoft.com/api/stac/v1',
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=['landsat-c2-l2'],
    bbox=[7.8, 46.3, 8.2, 46.7],
    datetime='2000-01-01/2023-12-31',
    query={'eo:cloud_cover': {'lt': 20}},
    max_items=200,
    sortby='datetime',
)

items = [i for i in search.items() if i.datetime.month in (6, 7, 8, 9)]
print(f'Found {len(items)} summer scenes')
for item in items:
    print(item.id, item.datetime.date(), f"{item.properties['eo:cloud_cover']:.1f}%")