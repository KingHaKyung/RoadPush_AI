# Commit Convention

당분간 팀 내 사용성을 우선해서 `[Type] English title` + 한국어 bullet summary 형식으로 작성한다.

## 기본 형식

```text
[Type] English title

- 변경 내용 1
- 변경 내용 2
- 변경 내용 3
```

## 작성 규칙

- 제목은 `[Type] English title` 형식으로 작성
- 제목은 영어 사용
- 본문 bullet은 한국어 사용
- 본문 bullet은 내부 클래스명, 함수명, 도구명만 나열하지 않고 변경 목적과 사용 방식을 풀어서 작성
- 고유명사나 코드 식별자를 써야 할 때도 처음 보는 사람이 이해할 수 있도록 한국어 설명을 함께 작성
- 제목과 본문은 Git 표준에 따라 빈 줄 1개로 구분
- bullet 사이에는 빈 줄을 절대 넣지 않음
- 여러 개의 -m 옵션을 사용할 때도 bullet 사이에 빈 줄이 생기지 않도록 하나의 본문 블록으로 작성
- 본문 bullet은 종결어미 없이 명사형/동사 어간형으로 작성
- 제목은 가능하면 60자 안팎으로 유지
- 본문 bullet은 필요한 경우에만 작성
- 한 커밋에는 하나의 의미 단위만 포함
- 작업 Slice와 커밋 단위는 동일하지 않으며 독립적으로 검토·검증·되돌릴 수 있는 의미 단위로 분리
- 테스트는 원칙적으로 검증 대상 구현과 같은 커밋에 포함
- Type은 변경된 파일 종류가 아니라 변경의 주된 목적을 기준으로 선택
- 실행 결과나 검증 내용이 있으면 마지막 bullet에 포함

## Summary 작성 원칙

summary는 구현 내부를 아는 사람보다 처음 보는 팀원이 변경 의도를 이해하는 것을 우선한다.

- CompactGraph 추가처럼 클래스명만 쓰지 않고 경로 탐색에 필요한 도로망을 가벼운 숫자 배열 구조로 변환처럼 목적과 방식을 설명
- selector 수정처럼 내부 용어만 쓰지 않고 출발지와 도착지에 맞춰 어떤 격자를 읽을지 고르는 정책 조정처럼 풀어서 설명
- RAG evidence 연결처럼 약어만 쓰지 않고 저장된 경로와 추천차단 결과를 AI 브리핑 근거로 읽도록 연결처럼 역할을 설명
- 영어 고유명사, 파일명, 클래스명은 필요할 때만 사용하고 한국어 설명을 함께 제공
- 검증을 수행했으면 마지막 bullet에 pytest 검증 통과, py_compile 검증 통과처럼 짧게 기록

### Summary 구체화 규칙

본문이 한국어 조사와 영어 식별자의 조합으로만 구성되지 않도록 다음 규칙을 적용한다.

- 모든 bullet은 처음 보는 사람이 이해할 수 있는 한국어 의미 설명을 중심으로 작성
- 클래스명, 함수명, 파일명, 라이브러리명만 나열한 bullet 금지
- 코드 식별자가 필요하면 같은 bullet 안에서 해당 식별자의 역할과 변경 목적을 한국어로 설명
- 무엇을 추가·수정했는지만 적지 않고 가능한 경우 왜 필요한지, 동작이 어떻게 달라지는지, 어떤 문제를 방지하는지를 설명
- 구현 세부사항보다 사용자 흐름, 데이터 흐름, 운영 안정성, 오류 방지 등 외부에서 관찰 가능한 변화를 우선
- 한 bullet에 모든 내용을 몰아넣지 않고 서로 다른 목적·동작·검증은 별도 bullet로 분리
- 검증 bullet은 명령어 또는 도구 이름만 적지 않고 어떤 범위를 확인했는지 함께 설명
- 영어 기술 용어는 정확한 검색이나 식별에 필요할 때만 유지하고, 나머지 문장은 자연스러운 한국어로 작성

나쁜 예:

```text
[Add] Add graph update service

- GraphChangeSet 추가
- CanonicalStore 연결
- SnapshotManager 적용
- pytest 통과
```

좋은 예:

```text
[Add] Add incremental graph updates

- 새 기사로 영향받은 사건과 이슈만 찾아 그래프 변경 범위 축소
- 변경 결과를 임시 저장소에서 검증한 뒤 정상 버전만 현재 그래프로 전환
- 증분 갱신과 전체 재구축 결과가 일치하는지 통합 테스트로 확인
```

코드 식별자가 필요한 좋은 예:

```text
- GraphChangeSet을 노드와 관계의 추가·갱신·비활성화 작업 목록으로 사용
```

다음과 같은 검증 bullet은 피한다.

```text
- pytest 통과
```

대신 검증한 범위를 함께 적는다.

```text
- 그래프 갱신 실패 시 이전 정상 버전이 유지되는지 통합 테스트로 확인
```

나쁜 예:

```text
[Add] Add compact runtime

- CompactGraph 추가

- selector 연결

- RAG evidence 수정
```

좋은 예:

```text
[Add] Add compact route runtime

- 경로 탐색에 필요한 도로망을 가벼운 숫자 배열 구조로 변환
- 출발지와 도착지 주변 격자만 읽어 전체 그래프 로딩 방지
- 추천차단 분석에서 같은 그래프 구조를 재사용할 수 있게 처리
```

## Type 목록

```text
[Init]      저장소 초기화, 프로젝트 골격 생성
[Add]       새 기능, 새 화면, 새 파일 추가
[Fix]       버그 수정, 오류 방지, 예외 처리
[Update]    기존 기능/설정/문서 내용 갱신
[Refactor]  동작 변경 없이 구조 개선
[Docs]      문서 추가/수정
[Test]      테스트, smoke test, 검증 코드 추가/수정
[Build]     Docker, 의존성, 배포/빌드 설정 변경
[Chore]     기타 유지보수, 포맷, 정리
```

## 예시

```text
[Init] Initialize PartsStudio v2 source repository

- 배포 대상 소스와 문서 첫 커밋
- 산출물, 모델 가중치, 로컬 설정 gitignore 제외
- GitHub 원격 저장소 연결
```

```text
[Add] Add trainer result analysis viewer

- selection run별 모델 성능 목록 표시
- prediction overlay와 label overlay 미리보기 추가
- 분석 zip 내보내기 경로 정리
```

```text
[Fix] Prevent stale SAM candidate application

- 후보 mask shape와 현재 이미지 크기 비교
- target 변경 후 잘못된 후보 적용 차단
- derived candidate 생성 경로에도 동일 검증 적용
```

```text
[Update] Update trainer worker Docker image tag

- sample inference 포함 이미지 태그로 변경
- Vast trainer docker example config 반영
- worker script compile smoke test 통과
```

```text
[Test] Add trainer package smoke test

- slot package 생성 경로 검증
- worker zip decode 테스트 추가
- py_compile 검증 통과
```

## Codex 작업 시 규칙

Codex가 커밋할 때는 이 문서를 기준으로 메시지를 작성한다.

### 커밋 메시지 입력 방식

bullet 사이에 빈 줄이 생기는 문제를 방지하기 위해 메시지 파일 사용을 우선한다.

```text
git commit --file <message-file>
```

`--file`의 단축 옵션은 대문자 `-F`다.

```text
git commit -F <message-file>
```

소문자 `-f`가 아니며 Git 옵션은 대소문자를 구분한다.

메시지 파일에는 제목, 제목과 본문 사이의 빈 줄 1개, 빈 줄 없이 연속된 bullet을 모두 한 번에 작성한다.

```text
[Add] Add incremental graph updates

- 새 기사로 영향받은 사건과 이슈만 찾아 그래프 변경 범위 축소
- 변경 결과를 검증한 뒤 정상 버전만 현재 그래프로 전환
- 증분 갱신 통합 테스트 통과
```

bullet마다 `git commit -m`을 반복해서 사용하면 안 된다.

다음 방식은 각 `-m`을 별도 문단으로 처리해 bullet 사이에 빈 줄을 삽입하므로 금지한다.

```text
git commit -m "[Add] Add incremental graph updates" -m "- 변경 내용 1" -m "- 변경 내용 2" -m "- 검증 결과"
```

메시지 파일을 사용하기 어려운 경우에만 다음 방식을 허용한다.

- 제목용 `-m` 1개
- 모든 bullet을 줄바꿈으로 연결한 단일 본문 `-m` 1개
- 본문용 `-m`을 둘 이상 사용하지 않음

커밋 후에는 실제 저장된 메시지를 다시 확인한다.

```text
git show -s --format=%B HEAD
```

확인 항목:

- 제목이 `[Type] English title` 형식인지 확인
- 제목과 본문 사이에 빈 줄이 정확히 1개인지 확인
- bullet 사이에 빈 줄이 없는지 확인
- 본문이 영어 식별자 나열이 아니라 한국어로 목적과 동작을 설명하는지 확인
- 검증을 수행했다면 마지막 bullet에 검증 범위와 결과가 있는지 확인

임시 메시지 파일은 저장소 밖 또는 gitignore 대상 위치에 만들고, UTF-8로 저장하며, 비밀정보를 포함하지 않고, 커밋 후 제거한다.

권장 흐름:

```text
1. 변경 작업 수행
2. 필요한 smoke test 실행
3. git diff 확인
4. 메시지 파일에 제목과 연속된 bullet 작성
5. git commit --file로 의미 단위별 커밋
6. 저장된 커밋 메시지 형식 재확인
7. 필요 시 push
```

커밋 요청 예시:

```text
지금 변경사항 이 컨벤션대로 커밋해줘
```
