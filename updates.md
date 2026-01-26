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
> version = "0.76.0"
< sdist = { url = "https://files.pythonhosted.org/packages/d7/7b/609eea5c54ae69b1a4a94169d4b0c86dc5c41b43509989913f6cdc61b81d/anthropic-0.74.1.tar.gz", hash = "sha256:04c087b2751385c524f6d332d066a913870e4de8b3e335fb0a0c595f1f88dc6e", size = 428981, upload-time = "2025-11-19T22:17:31.533Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/6e/be/d11abafaa15d6304826438170f7574d750218f49a106c54424a40cef4494/anthropic-0.76.0.tar.gz", hash = "sha256:e0cae6a368986d5cf6df743dfbb1b9519e6a9eee9c6c942ad8121c0b34416ffe", size = 495483, upload-time = "2026-01-13T18:41:14.908Z" }
<     { url = "https://files.pythonhosted.org/packages/dd/45/6b18d0692302b8cbc01a10c35b43953d3c4172fbd4f83337b8ed21a8eaa4/anthropic-0.74.1-py3-none-any.whl", hash = "sha256:b07b998d1cee7f41d9f02530597d7411672b362cc2417760a40c0167b81c6e65", size = 371473, upload-time = "2025-11-19T22:17:29.998Z" },
>     { url = "https://files.pythonhosted.org/packages/e5/70/7b0fd9c1a738f59d3babe2b4212031c34ab7d0fda4ffef15b58a55c5bcea/anthropic-0.76.0-py3-none-any.whl", hash = "sha256:81efa3113901192af2f0fe977d3ec73fdadb1e691586306c4256cd6d5ccc331c", size = 390309, upload-time = "2026-01-13T18:41:13.483Z" },
< version = "4.11.0"
> version = "4.12.1"
<     { name = "sniffio" },
< sdist = { url = "https://files.pythonhosted.org/packages/c6/78/7d432127c41b50bccba979505f272c16cbcadcc33645d5fa3a738110ae75/anyio-4.11.0.tar.gz", hash = "sha256:82a8d0b81e318cc5ce71a5f1f8b5c4e63619620b63141ef8c995fa0db95a57c4", size = 219094, upload-time = "2025-09-23T09:19:12.58Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/96/f0/5eb65b2bb0d09ac6776f2eb54adee6abe8228ea05b20a5ad0e4945de8aac/anyio-4.12.1.tar.gz", hash = "sha256:41cfcc3a4c85d3f05c932da7c26d0201ac36f72abd4435ba90d0464a3ffed703", size = 228685, upload-time = "2026-01-06T11:45:21.246Z" }
<     { url = "https://files.pythonhosted.org/packages/15/b3/9b1a8074496371342ec1e796a96f99c82c945a339cd81a8e73de28b4cf9e/anyio-4.11.0-py3-none-any.whl", hash = "sha256:0287e96f4d26d4149305414d4e3bc32f0dcd0862365a4bddea19d7a1ec38c4fc", size = 109097, upload-time = "2025-09-23T09:19:10.601Z" },
>     { url = "https://files.pythonhosted.org/packages/38/0e/27be9fdef66e72d64c0cdc3cc2823101b80585f8119b5c112c2e8f5f7dab/anyio-4.12.1-py3-none-any.whl", hash = "sha256:d405828884fc140aa80a3c667b8beed277f1dfedec42ba031bd6ac3db606ab6c", size = 113592, upload-time = "2026-01-06T11:45:19.497Z" },
< version = "4.0.2"
> version = "4.0.3"
< sdist = { url = "https://files.pythonhosted.org/packages/b7/22/97df040e15d964e592d3a180598ace67e91b7c559d8298bdb3c949dc6e42/astroid-4.0.2.tar.gz", hash = "sha256:ac8fb7ca1c08eb9afec91ccc23edbd8ac73bb22cbdd7da1d488d9fb8d6579070", size = 405714, upload-time = "2025-11-09T21:21:18.373Z" }
> sdist = { url = "https://files.pythonhosted.org/packages/a1/ca/c17d0f83016532a1ad87d1de96837164c99d47a3b6bbba28bd597c25b37a/astroid-4.0.3.tar.gz", hash = "sha256:08d1de40d251cc3dc4a7a12726721d475ac189e4e583d596ece7422bc176bda3", size = 406224, upload-time = "2026-01-03T22:14:26.096Z" }
<     { url = "https://files.pythonhosted.org/packages/93/ac/a85b4bfb4cf53221513e27f33cc37ad158fce02ac291d18bee6b49ab477d/astroid-4.0.2-py3-none-any.whl", hash = "sha256:d7546c00a12efc32650b19a2bb66a153883185d3179ab0d4868086f807338b9b", size = 276354, upload-time = "2025-11-09T21:21:16.54Z" },
>     { url = "https://files.pythonhosted.org/packages/ce/66/686ac4fc6ef48f5bacde625adac698f41d5316a9753c2b20bb0931c9d4e2/astroid-4.0.3-py3-none-any.whl", hash = "sha256:864a0a34af1bd70e1049ba1e61cee843a7252c826d97825fcee9b2fcbd9e1b14", size = 276443, upload-time = "2026-01-03T22:14:24.412Z" },
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
>     "python_full_version == '3.11.*' and sys_platform == 'emscripten'",
>     "python_full_version >= '3.12' and sys_platform != 'emscripten' and sys_platform != 'win32'",
>     "python_full_version == '3.11.*' and sys_platform != 'emscripten' and sys_platform != 'win32'",
< version = "3.3.0"
> version = "3.4.0"
< sdist = { url = "https://files.pythonhosted.org/packages/e2/f9/516671861cf190bda37f6afa696d8a6a6ac593f23d8cf198e16faca044f5/dash-3.3.0.tar.gz", hash = "sha256:eaaa7a671540b5e1db8066f4966d0277d21edc2c7acdaec2fd6d198366a8b0df", size = 7579436, upload-time = "2025-11-12T15:51:54.919Z" }
```
