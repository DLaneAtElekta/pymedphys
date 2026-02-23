## Dependency Updates

```diff
<     "python_full_version >= '3.12'",
<     "python_full_version == '3.11.*'",
>     "python_full_version >= '3.12' and sys_platform == 'win32'",
>     "python_full_version == '3.11.*' and sys_platform == 'win32'",
>     "python_full_version >= '3.12' and sys_platform == 'emscripten'",
>     "python_full_version == '3.11.*' and sys_platform == 'emscripten'",
>     "python_full_version >= '3.12' and sys_platform != 'emscripten' and sys_platform != 'win32'",
>     "python_full_version == '3.11.*' and sys_platform != 'emscripten' and sys_platform != 'win32'",
< version = "0.74.1"
> version = "0.83.0"
< sdist = { url = "https://files.pythonhosted.org/packages/d7/7b/609eea5c54ae69b1a4a94169d4b0c86dc5c41b43509989913f6cdc61b81d/anthropic-0.74.1.tar.gz", hash = "sha256:04c087b2751385c524f6d332d066a913870e4de8b3e335fb0a0c595f1f88dc6e", size = 428981, upload-time = "2025-11-19T22:17:31.533Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/db/e5/02cd2919ec327b24234abb73082e6ab84c451182cc3cc60681af700f4c63/anthropic-0.83.0.tar.gz", hash = "sha256:a8732c68b41869266c3034541a31a29d8be0f8cd0a714f9edce3128b351eceb4", size = 534058, upload-time = "2026-02-19T19:26:38.904Z" }
<     { url = "https://files.pythonhosted.org/packages/dd/45/6b18d0692302b8cbc01a10c35b43953d3c4172fbd4f83337b8ed21a8eaa4/anthropic-0.74.1-py3-none-any.whl", hash = "sha256:b07b998d1cee7f41d9f02530597d7411672b362cc2417760a40c0167b81c6e65", size = 371473, upload-time = "2025-11-19T22:17:29.998Z" },
>     { url = "https://files.pythonhosted.org/packages/5f/75/b9d58e4e2a4b1fc3e75ffbab978f999baf8b7c4ba9f96e60edb918ba386b/anthropic-0.83.0-py3-none-any.whl", hash = "sha256:f069ef508c73b8f9152e8850830d92bd5ef185645dbacf234bb213344a274810", size = 456991, upload-time = "2026-02-19T19:26:40.114Z" },
< version = "4.11.0"
> version = "4.12.1"
<     { name = "sniffio" },
< sdist = { url = "https://files.pythonhosted.org/packages/c6/78/7d432127c41b50bccba979505f272c16cbcadcc33645d5fa3a738110ae75/anyio-4.11.0.tar.gz", hash = "sha256:82a8d0b81e318cc5ce71a5f1f8b5c4e63619620b63141ef8c995fa0db95a57c4", size = 219094, upload-time = "2025-09-23T09:19:12.58Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/96/f0/5eb65b2bb0d09ac6776f2eb54adee6abe8228ea05b20a5ad0e4945de8aac/anyio-4.12.1.tar.gz", hash = "sha256:41cfcc3a4c85d3f05c932da7c26d0201ac36f72abd4435ba90d0464a3ffed703", size = 228685, upload-time = "2026-01-06T11:45:21.246Z" }
<     { url = "https://files.pythonhosted.org/packages/15/b3/9b1a8074496371342ec1e796a96f99c82c945a339cd81a8e73de28b4cf9e/anyio-4.11.0-py3-none-any.whl", hash = "sha256:0287e96f4d26d4149305414d4e3bc32f0dcd0862365a4bddea19d7a1ec38c4fc", size = 109097, upload-time = "2025-09-23T09:19:10.601Z" },
>     { url = "https://files.pythonhosted.org/packages/38/0e/27be9fdef66e72d64c0cdc3cc2823101b80585f8119b5c112c2e8f5f7dab/anyio-4.12.1-py3-none-any.whl", hash = "sha256:d405828884fc140aa80a3c667b8beed277f1dfedec42ba031bd6ac3db606ab6c", size = 113592, upload-time = "2026-01-06T11:45:19.497Z" },
< version = "4.0.2"
> version = "4.0.4"
< sdist = { url = "https://files.pythonhosted.org/packages/b7/22/97df040e15d964e592d3a180598ace67e91b7c559d8298bdb3c949dc6e42/astroid-4.0.2.tar.gz", hash = "sha256:ac8fb7ca1c08eb9afec91ccc23edbd8ac73bb22cbdd7da1d488d9fb8d6579070", size = 405714, upload-time = "2025-11-09T21:21:18.373Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/07/63/0adf26577da5eff6eb7a177876c1cfa213856be9926a000f65c4add9692b/astroid-4.0.4.tar.gz", hash = "sha256:986fed8bcf79fb82c78b18a53352a0b287a73817d6dbcfba3162da36667c49a0", size = 406358, upload-time = "2026-02-07T23:35:07.509Z" }
<     { url = "https://files.pythonhosted.org/packages/93/ac/a85b4bfb4cf53221513e27f33cc37ad158fce02ac291d18bee6b49ab477d/astroid-4.0.2-py3-none-any.whl", hash = "sha256:d7546c00a12efc32650b19a2bb66a153883185d3179ab0d4868086f807338b9b", size = 276354, upload-time = "2025-11-09T21:21:16.54Z" },
>     { url = "https://files.pythonhosted.org/packages/b0/cf/1c5f42b110e57bc5502eb80dbc3b03d256926062519224835ef08134f1f9/astroid-4.0.4-py3-none-any.whl", hash = "sha256:52f39653876c7dec3e3afd4c2696920e05c83832b9737afc21928f2d2eb7a753", size = 276445, upload-time = "2026-02-07T23:35:05.344Z" },
< version = "2.17.0"
> version = "2.18.0"
< sdist = { url = "https://files.pythonhosted.org/packages/7d/6b/d52e42361e1aa00709585ecc30b3f9684b3ab62530771402248b1b1d6240/babel-2.17.0.tar.gz", hash = "sha256:0c54cffb19f690cdcc52a3b50bcbf71e07a808d1c80d549f2459b9d2cf0afb9d", size = 9951852, upload-time = "2025-02-01T15:17:41.026Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/7d/b2/51899539b6ceeeb420d40ed3cd4b7a40519404f9baf3d4ac99dc413a834b/babel-2.18.0.tar.gz", hash = "sha256:b80b99a14bd085fcacfa15c9165f651fbb3406e66cc603abf11c5750937c992d", size = 9959554, upload-time = "2026-02-01T12:30:56.078Z" }
<     { url = "https://files.pythonhosted.org/packages/b7/b8/3fe70c75fe32afc4bb507f75563d39bc5642255d1d94f1f23604725780bf/babel-2.17.0-py3-none-any.whl", hash = "sha256:4d0b53093fdfb4b21c92b5213dba5a1b23885afa8383709427046b21c366e5f2", size = 10182537, upload-time = "2025-02-01T15:17:37.39Z" },
>     { url = "https://files.pythonhosted.org/packages/77/f5/21d2de20e8b8b0408f0681956ca2c69f1320a3848ac50e6e7f39c6159675/babel-2.18.0-py3-none-any.whl", hash = "sha256:e2b422b277c2b9a9630c1d7903c2a00d0830c409c59ac8cae9081c92f1aeba35", size = 10196845, upload-time = "2026-02-01T12:30:53.445Z" },
< version = "4.14.2"
> version = "4.14.3"
< sdist = { url = "https://files.pythonhosted.org/packages/77/e9/df2358efd7659577435e2177bfa69cba6c33216681af51a707193dec162a/beautifulsoup4-4.14.2.tar.gz", hash = "sha256:2a98ab9f944a11acee9cc848508ec28d9228abfd522ef0fad6a02a72e0ded69e", size = 625822, upload-time = "2025-09-29T10:05:42.613Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/c3/b0/1c6a16426d389813b48d95e26898aff79abbde42ad353958ad95cc8c9b21/beautifulsoup4-4.14.3.tar.gz", hash = "sha256:6292b1c5186d356bba669ef9f7f051757099565ad9ada5dd630bd9de5fa7fb86", size = 627737, upload-time = "2025-11-30T15:08:26.084Z" }
<     { url = "https://files.pythonhosted.org/packages/94/fe/3aed5d0be4d404d12d36ab97e2f1791424d9ca39c2f754a6285d59a3b01d/beautifulsoup4-4.14.2-py3-none-any.whl", hash = "sha256:5ef6fa3a8cbece8488d66985560f97ed091e22bbc4e9c2338508a9d5de6d4515", size = 106392, upload-time = "2025-09-29T10:05:43.771Z" },
>     { url = "https://files.pythonhosted.org/packages/1a/39/47f9197bdd44df24d67ac8893641e16f386c984a0619ef2ee4c51fbbc019/beautifulsoup4-4.14.3-py3-none-any.whl", hash = "sha256:0918bfe44902e6ad8d57732ba310582e98da931428d231a5ecb9e7c703a735bb", size = 107721, upload-time = "2025-11-30T15:08:24.087Z" },
< version = "2025.11.12"
> version = "2026.1.4"
< sdist = { url = "https://files.pythonhosted.org/packages/a2/8c/58f469717fa48465e4a50c014a0400602d3c437d7c0c468e17ada824da3a/certifi-2025.11.12.tar.gz", hash = "sha256:d8ab5478f2ecd78af242878415affce761ca6bc54a22a27e026d7c25357c3316", size = 160538, upload-time = "2025-11-12T02:54:51.517Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/e0/2d/a891ca51311197f6ad14a7ef42e2399f36cf2f9bd44752b3dc4eab60fdc5/certifi-2026.1.4.tar.gz", hash = "sha256:ac726dd470482006e014ad384921ed6438c457018f4b3d204aea4281258b2120", size = 154268, upload-time = "2026-01-04T02:42:41.825Z" }
<     { url = "https://files.pythonhosted.org/packages/70/7d/9bc192684cea499815ff478dfcdc13835ddf401365057044fb721ec6bddb/certifi-2025.11.12-py3-none-any.whl", hash = "sha256:97de8790030bbd5c2d96b7ec782fc2f7820ef8dba6db909ccf95449f2d062d4b", size = 159438, upload-time = "2025-11-12T02:54:49.735Z" },
>     { url = "https://files.pythonhosted.org/packages/e6/ad/3cc14f097111b4de0040c83a525973216457bbeeb63739ef1ed275c1c021/certifi-2026.1.4-py3-none-any.whl", hash = "sha256:9943707519e4add1115f44c2bc244f782c0249876bf51b6599fee1ffbedd685c", size = 152900, upload-time = "2026-01-04T02:42:40.15Z" },
<     "python_full_version >= '3.12'",
<     "python_full_version == '3.11.*'",
>     "python_full_version >= '3.12' and sys_platform == 'win32'",
>     "python_full_version == '3.11.*' and sys_platform == 'win32'",
>     "python_full_version >= '3.12' and sys_platform == 'emscripten'",
```
