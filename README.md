### Dataset 
“disparbud-od_15387_jml_ptns_obyek_daya_tarik_wisata_odtw__jenis_kabup_v2_data.csv” berisi informasi tentang jumlah daya tarik wisata di berbagai kabupaten dan kota di Indonesia, khususnya di Provinsi Jawa Barat.
link: https://opendata.jabarprov.go.id/id/dataset/jumlah-potensi-obyek-daya-tarik-wisata-odtw-berdasarkan-jenis-dan-kabupatenkota-di-jawa-barat

### Kolom Utama:
- **id** : Pengenal unik untuk setiap entri.
- **kode_provinsi** : Kode yang mewakili provinsi (dalam hal ini Jawa Barat diwakili oleh angka 32).
- **nama_provinsi** : Nama provinsi (JAWA BARAT).
- **kode_kabupaten_kota** : Kode yang mewakili distrik atau kota tertentu.
- **nama_kabupaten_kota** : Nama kabupaten atau kota dalam provinsi.
- **jenis_odtw** : Jenis objek wisata (misalnya alam, budaya, dan lain-lain).
- **jumlah_odtw** : Jumlah objek wisata dengan tipe tertentu di kabupaten/kota tersebut.
- **satuan** : Satuan ukuran, yang menunjukkan bahwa hitungannya berdasarkan jumlah lokasi.
- **tahun** : Tahun data dikumpulkan.
Contoh Entri:
Misalnya saja pada tahun 2014, Kabupaten Bogor memiliki 38 objek wisata alam (ALAM) .

Kumpulan data ini berguna untuk menganalisis tren pariwisata di berbagai wilayah dan jenis objek wisata, yang dapat membantu dalam upaya perencanaan dan pengembangan pariwisata regional.

## Notebook ini membahas tentang:
- Pemuatan dan pemahaman himpunan data,
- Penanganan nilai yang hilang dan outlier,
- Visualisasi aspek utama,
- Penerapan model regresi linier, dan
- Pengelompokan data dengan K-Means.
