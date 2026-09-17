const demoCandidates={
  shiba:[
    {name:'갈색 시바 믹스 · 수컷',place:'경기 화성시 보호 공고',date:'발견 2일 전',score:94,why:'갈색 털 · 뾰족한 귀 · 체형'},
    {name:'황갈색 진도 믹스 · 암컷',place:'경기 수원시 보호 공고',date:'발견 4일 전',score:87,why:'얼굴 윤곽 · 귀 모양 · 인접 지역'},
    {name:'갈색 중형 믹스견 · 수컷',place:'경기 오산시 보호 공고',date:'발견 6일 전',score:81,why:'털색 · 체형 · 날짜 범위'},
    {name:'크림색 시바 믹스 · 미상',place:'서울 관악구 보호 공고',date:'발견 8일 전',score:74,why:'꼬리 모양 · 얼굴 윤곽'}
  ],
  dachshund:[
    {name:'검정·갈색 닥스훈트 · 수컷',place:'경기 안성시 보호 공고',date:'발견 1일 전',score:95,why:'긴 몸 · 짧은 다리 · 털색'},
    {name:'갈색 닥스훈트 믹스 · 암컷',place:'경기 평택시 보호 공고',date:'발견 3일 전',score:88,why:'체형 · 귀 모양 · 인접 지역'},
    {name:'검정 소형 믹스견 · 수컷',place:'경기 용인시 보호 공고',date:'발견 5일 전',score:79,why:'검정 털 · 낮은 체형'},
    {name:'갈색 단모 소형견 · 미상',place:'서울 송파구 보호 공고',date:'발견 7일 전',score:71,why:'털색 · 얼굴 윤곽'}
  ]
};
const demos=[
  // Every prepared demo has at least two available photos for one notice:
  // popfile1 is the query and popfile2 remains as a cross-view candidate.
  {id:'shiba',label:'예시 1',name:'갈색 시바',image:'./assets/demo-shiba.jpg',feature:'갈색 털, 뾰족한 귀, 중형견',location:'경기 화성시',date:'2026-09-08',animalId:'441553202602531',availablePhotos:2},
  {id:'dachshund',label:'예시 2',name:'닥스훈트',image:'./assets/demo-dachshund.jpg',feature:'검정·갈색 털, 긴 몸, 짧은 다리',location:'경기 안성시',date:'2026-09-08',animalId:'441408202601716',availablePhotos:2}
];
const results=document.querySelector('#results');
const button=document.querySelector('#search');
let apiBase='',live=false,selectedDemoId='shiba';
const safe=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function renderStatic(list){results.innerHTML=list.map((x,i)=>`<article class="candidate"><div class="dog" aria-hidden="true">🐕</div><div><h3>${i+1}. ${x.name}</h3><p>${x.place} · ${x.date}</p><p>근거: ${x.why}</p></div><div class="score"><strong>${x.score}%</strong><small>예시 유사도</small></div></article>`).join('')}
function renderLive(list){results.innerHTML=list.map((x,i)=>`<article class="candidate"><img class="dog result-photo" src="${apiBase}/api/public/animals/${encodeURIComponent(x.animal_id)}/image?slot=${encodeURIComponent(x.image_slot||'popfile1')}" alt="${safe(x.kind_name||'보호 공고')} 후보 사진"><div><h3>${i+1}. ${safe(x.kind_name||'품종 미상')} · ${safe(x.sex||'성별 미상')}</h3><p>${safe(x.happen_place||'발견 장소 미상')} · ${safe(x.happen_date||'날짜 미상')}</p><p>근거: ${safe(x.rationale||'이미지 유사도')}</p></div><div class="score"><strong>${(Number(x.final_score||0)*100).toFixed(1)}%</strong><small>후보 점수</small></div></article>`).join('')}
function renderDemos(){document.querySelector('#demoOptions').innerHTML=demos.filter(x=>x.availablePhotos>=2).map(x=>`<button class="demo-option" type="button" data-demo="${x.id}" aria-pressed="false"><img src="${x.image}" alt=""><span><small>${x.label}</small><b>${x.name}</b></span></button>`).join('')}
async function selectDemo(id){const demo=demos.find(x=>x.id===id);if(!demo)return;selectedDemoId=id;const response=await fetch(demo.image),blob=await response.blob(),file=new File([blob],`${demo.id}.jpg`,{type:blob.type||'image/jpeg'}),transfer=new DataTransfer();transfer.items.add(file);document.querySelector('#photo').files=transfer.files;document.querySelector('#features').value=demo.feature;document.querySelector('#location').value=demo.location;document.querySelector('#date').value=demo.date;const preview=document.querySelector('#preview');preview.src=demo.image;preview.hidden=false;document.querySelector('#dropPrompt').hidden=true;document.querySelectorAll('.demo-option').forEach(node=>{const selected=node.dataset.demo===id;node.classList.toggle('selected',selected);node.setAttribute('aria-pressed',String(selected))});if(live){document.querySelector('#resultState').textContent=`${demo.name} 예시가 준비되었습니다. ‘유사 후보 살펴보기’를 눌러보세요.`}else{renderStatic(demoCandidates[id]);document.querySelector('#resultState').textContent=`${demo.name} 합성 예시 후보로 업데이트했습니다.`}}
async function fetchTimed(url,options={},milliseconds=12000){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),milliseconds);try{return await fetch(url,{...options,signal:controller.signal})}finally{clearTimeout(timer)}}
async function initialize(){try{const config=await fetch(`./data/api-config.json?ts=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),url=new URL(config.apiBase);if(url.protocol!=='https:'||url.pathname!=='/'||url.username||url.password)throw new Error();apiBase=url.origin;const response=await fetchTimed(`${apiBase}/api/public/health`),health=await response.json();if(!response.ok||health.status!=='ok'||health.data_classification!=='public_notice_demo')throw new Error();live=true;document.querySelector('#connectionStatus').innerHTML='<i></i> Mac 검색 서버 연결';document.querySelector('#resultState').textContent='사진을 선택하면 실제 공개 공고를 검색합니다.'}catch{live=false;document.querySelector('#connectionStatus').innerHTML='<i class="offline"></i> 합성 데모 모드';document.querySelector('#resultState').textContent='Mac 서버가 꺼져 있어 예시 후보를 표시합니다.'}}
renderStatic(demoCandidates.shiba);
renderDemos();
document.querySelector('#demoOptions').addEventListener('click',event=>{const option=event.target.closest('.demo-option');if(option)selectDemo(option.dataset.demo)});
document.querySelector('#photo').addEventListener('change',event=>{const file=event.target.files[0];if(!file)return;selectedDemoId='';const preview=document.querySelector('#preview');preview.src=URL.createObjectURL(file);preview.hidden=false;document.querySelector('#dropPrompt').hidden=true});
button.addEventListener('click',async()=>{const file=document.querySelector('#photo').files[0],feature=document.querySelector('#features').value.trim(),location=document.querySelector('#location').value.trim(),date=document.querySelector('#date').value;button.disabled=true;button.firstChild.textContent=live?'실제 공고 검색 중…':'합성 후보 정렬 중…';try{if(live){if(!file)throw new Error('실제 검색에는 사진이 필요합니다.');const form=new FormData();form.append('file',file);form.append('top_k','5');form.append('feature_text',feature);form.append('location_text',location);form.append('missing_date',date);form.append('exclude_exact_image',String(Boolean(selectedDemoId)));const response=await fetchTimed(`${apiBase}/api/public/search`,{method:'POST',body:form},30000),data=await response.json();if(!response.ok)throw new Error(data.detail||'검색 실패');renderLive(data.items);document.querySelector('#resultState').textContent=selectedDemoId?'동일 사진을 제외하고 같은 개체의 다른 사진을 포함한 후보입니다.':'Mac 검색 엔진이 반환한 실제 공개 공고 후보입니다.'}else if(selectedDemoId){renderStatic(demoCandidates[selectedDemoId]);document.querySelector('#resultState').textContent='선택한 합성 예시의 후보 결과입니다.'}else{renderStatic(demoCandidates.shiba);document.querySelector('#resultState').textContent='Mac 서버가 꺼져 있어 사용자 사진 대신 합성 예시 후보를 표시합니다.'}}catch(error){document.querySelector('#resultState').textContent=error.message||'검색 서버와 통신하지 못했습니다.'}finally{button.disabled=false;button.firstChild.textContent='유사 후보 살펴보기 ';document.querySelector('.results-panel').scrollIntoView({behavior:'smooth',block:'start'})}});
async function startDemo(){await initialize();await selectDemo('shiba')}
startDemo();
