# Licensing and third-party notices

This repository combines code from several sources. The license in the agent
subdirectory is not a license for the entire repository.

## Hisss simulator: upstream permission still unresolved

`src/hisss` is derived from [ymahlau/hisss](https://github.com/ymahlau/hisss).
The upstream project metadata credits Yannik Mahlau, Robin Schmöcker, and
Christoph Schnell. Those credits are retained in `pyproject.toml`.

As checked on 2026-09-19, the upstream main branch contains no license file and
GitHub reports no detected license. No separate permission from the upstream
rights holders has been recorded here. Before publishing this derivative, obtain
and record the applicable license or explicit redistribution permission. This
notice does not grant rights to upstream code or relicense it.

## Bundled ALGLIB

`src/hisss/cpp/alglib` contains ALGLIB 3.20.0, copyright Sergey Bochkanov
(ALGLIB project), generated on 2022-12-19. Its source headers specify the GNU
General Public License, version 2 or (at your option) any later version
(`GPL-2.0-or-later`). Original copyright and license headers are retained.

The complete version 2 license text is in
[`LICENSES/GPL-2.0-or-later.txt`](LICENSES/GPL-2.0-or-later.txt), obtained from
<https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt>.

The simulator's CMake build links ALGLIB into `liblink`. Distribution of the
combined library must account for the GPL terms as well as the upstream Hisss
permission above. Including the GPL text alone does not resolve missing rights
to other code. Source files and build instructions are included in this
repository; binary releases must also satisfy the corresponding source
requirements of the applicable GPL version.

## Battlesnake Blackout starter

`bs-blackout-starter` is based on
[l-berg/battlesnake-blackout-starter](https://github.com/l-berg/battlesnake-blackout-starter).
Its original MIT license, copyright (c) 2026 Lukas Berg, is preserved verbatim in
[`bs-blackout-starter/LICENSE`](bs-blackout-starter/LICENSE).
The starter in turn credits
[Battlesnake's Python starter](https://github.com/BattlesnakeOfficial/starter-snake-python)
for its API and design.

The starter's MIT notice does not change the licenses of the separately bundled
simulator or ALGLIB. No repository-wide license is asserted here while upstream
simulator permissions remain unresolved.
