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
  {id:'shiba',label:'예시 1',name:'갈색 시바',image:'./assets/demo-shiba.jpg',breed:'시바',feature:'갈색 털, 뾰족한 귀, 중형견',sido:'경기도',sigungu:'화성시',date:'2026-09-08',animalId:'441553202602531',availablePhotos:2},
  {id:'dachshund',label:'예시 2',name:'닥스훈트',image:'./assets/demo-dachshund.jpg',breed:'닥스훈트',feature:'검정·갈색 털, 긴 몸, 짧은 다리',sido:'경기도',sigungu:'안성시',date:'2026-09-08',animalId:'441408202601716',availablePhotos:2}
];
const MAX_PHOTOS=5,PAGE_SIZE=10,SERVER_TOP_K=20,UNKNOWN_BREED='모름',OTHER_SIGUNGU='__other__';
const $=selector=>document.querySelector(selector);
const results=$('#results');
const button=$('#search');
let apiBase='',live=false,selectedDemoId='shiba',photos=[],regions=[],ranked=[],page=0;
const safe=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// ---- 사진: 여러 장, 잘리지 않게 ----
function renderPhotos(){
  const tiles=photos.map((photo,i)=>`<figure class="photo-tile"><img src="${photo.url}" alt="선택한 사진 ${i+1}"><button type="button" class="photo-remove" data-index="${i}" aria-label="사진 ${i+1} 삭제">×</button></figure>`);
  if(photos.length<MAX_PHOTOS)tiles.push(`<label class="photo-add" for="photo"><b>＋</b><span>${photos.length?'사진 추가':'사진 선택'}</span><small>${photos.length}/${MAX_PHOTOS}</small></label>`);
  $('#photoGrid').innerHTML=tiles.join('');
  $('#photoGrid').classList.toggle('empty',!photos.length);
}
function addPhotos(files){
  const images=[...files].filter(file=>file.type.startsWith('image/'));
  const room=MAX_PHOTOS-photos.length;
  images.slice(0,room).forEach(file=>photos.push({file,url:URL.createObjectURL(file)}));
  if(images.length>room)$('#resultState').textContent=`사진은 최대 ${MAX_PHOTOS}장까지 넣을 수 있어요.`;
  renderPhotos();
}
function clearPhotos(){photos.forEach(photo=>URL.revokeObjectURL(photo.url));photos=[]}

// ---- 품종 / 지역 선택지 (animal.go.kr API 표기 그대로) ----
async function loadOptions(){
  try{
    const options=await fetch('./data/options.json',{cache:'no-store'}).then(r=>r.json());
    regions=options.regions||[];
    $('#breedList').innerHTML=[UNKNOWN_BREED,...(options.breeds||[])].map(x=>`<option value="${safe(x)}"></option>`).join('');
    $('#sido').innerHTML='<option value="">시·도 선택</option>'+regions.map(r=>`<option value="${safe(r.sido)}">${safe(r.sido)}</option>`).join('');
  }catch{
    // 선택지를 못 불러오면 직접 입력으로 대체한다.
    $('#sido').innerHTML='<option value="">시·도 목록을 불러오지 못했어요</option>';
    $('#locationDetail').placeholder='실종 장소 직접 입력 예: 서울특별시 동작구 사당동';
  }
}
function fillSigungu(sido){
  const select=$('#sigungu'),region=regions.find(r=>r.sido===sido),list=region?region.sigungu:[];
  select.innerHTML=`<option value="">${list.length?'시·군·구 전체':'시·군·구 없음'}</option>`+list.map(x=>`<option value="${safe(x)}">${safe(x)}</option>`).join('')+(sido?`<option value="${OTHER_SIGUNGU}">목록에 없어요 (직접 입력)</option>`:'');
  select.disabled=!sido;
  updateDetailPlaceholder();
}
function updateDetailPlaceholder(){
  $('#locationDetail').placeholder=$('#sigungu').value===OTHER_SIGUNGU?'시·군·구와 동네 입력 예: 동작구 사당동':'읍·면·동 (선택) 예: 사당동';
}
function locationText(){
  const sigungu=$('#sigungu').value;
  return [$('#sido').value,sigungu===OTHER_SIGUNGU?'':sigungu,$('#locationDetail').value.trim()].filter(Boolean).join(' ');
}
function featureText(){
  const breed=$('#breed').value.trim();
  return [breed===UNKNOWN_BREED?'':breed,$('#features').value.trim()].filter(Boolean).join(', ');
}

// ---- 결과: 10개씩 보기 ----
function renderStatic(list){ranked=[];$('#pager').hidden=true;results.innerHTML=list.map((x,i)=>`<article class="candidate"><div class="dog" aria-hidden="true">🐕</div><div><h3>${i+1}. ${x.name}</h3><p>${x.place} · ${x.date}</p><p>근거: ${x.why}</p></div><div class="score"><strong>${x.score}%</strong><small>예시 유사도</small></div></article>`).join('')}
function renderLivePage(){
  const start=page*PAGE_SIZE,list=ranked.slice(start,start+PAGE_SIZE);
  results.innerHTML=list.map((x,i)=>`<article class="candidate"><img class="dog result-photo" src="${apiBase}/api/public/animals/${encodeURIComponent(x.animal_id)}/image?slot=${encodeURIComponent(x.image_slot||'popfile1')}" alt="${safe(x.kind_name||'보호 공고')} 후보 사진" loading="lazy"><div><h3>${start+i+1}. ${safe(x.kind_name||'품종 미상')} · ${safe(x.sex||'성별 미상')}</h3><p>${safe(x.happen_place||'발견 장소 미상')} · ${safe(x.happen_date||'날짜 미상')}</p><p>근거: ${safe(x.rationale||'이미지 유사도')}${x.photoHits>1?` · 사진 ${x.photoHits}장에서 발견`:''}</p></div><div class="score"><strong>${(Number(x.final_score||0)*100).toFixed(1)}%</strong><small>후보 점수</small></div></article>`).join('');
  const pages=Math.ceil(ranked.length/PAGE_SIZE),last=page>=pages-1;
  $('#pager').hidden=pages<=1;
  $('#pageInfo').textContent=`${start+1}–${start+list.length}위 / 전체 ${ranked.length}개`;
  $('#reroll').firstChild.textContent=last?'처음 10개 다시 보기 ':`다음 후보 ${Math.min(PAGE_SIZE,ranked.length-start-PAGE_SIZE)}개 `;
}
// 사진마다 검색한 결과를 공고 단위로 합친다: 가장 높은 점수를 쓰고, 몇 장에서 나왔는지 센다.
function mergeResults(lists){
  const byAnimal=new Map();
  lists.flat().forEach(item=>{
    const prev=byAnimal.get(item.animal_id);
    if(!prev)byAnimal.set(item.animal_id,{...item,photoHits:1});
    else{prev.photoHits+=1;if(Number(item.final_score)>Number(prev.final_score))Object.assign(prev,item,{photoHits:prev.photoHits})}
  });
  return [...byAnimal.values()].sort((a,b)=>Number(b.final_score)-Number(a.final_score)||b.photoHits-a.photoHits);
}

// ---- 데모 / 서버 ----
function renderDemos(){$('#demoOptions').innerHTML=demos.filter(x=>x.availablePhotos>=2).map(x=>`<button class="demo-option" type="button" data-demo="${x.id}" aria-pressed="false"><img src="${x.image}" alt=""><span><small>${x.label}</small><b>${x.name}</b></span></button>`).join('')}
async function selectDemo(id){
  const demo=demos.find(x=>x.id===id);if(!demo)return;
  selectedDemoId=id;
  const blob=await fetch(demo.image).then(r=>r.blob());
  clearPhotos();addPhotos([new File([blob],`${demo.id}.jpg`,{type:blob.type||'image/jpeg'})]);
  $('#breed').value=demo.breed;$('#features').value=demo.feature;$('#date').value=demo.date;
  $('#sido').value=demo.sido;fillSigungu($('#sido').value);$('#sigungu').value=demo.sigungu;$('#locationDetail').value='';updateDetailPlaceholder();
  document.querySelectorAll('.demo-option').forEach(node=>{const selected=node.dataset.demo===id;node.classList.toggle('selected',selected);node.setAttribute('aria-pressed',String(selected))});
  if(live){$('#resultState').textContent=`${demo.name} 예시가 준비되었습니다. ‘유사 후보 살펴보기’를 눌러보세요.`}else{renderStatic(demoCandidates[id]);$('#resultState').textContent=`${demo.name} 합성 예시 후보로 업데이트했습니다.`}
}
async function fetchTimed(url,options={},milliseconds=12000){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),milliseconds);try{return await fetch(url,{...options,signal:controller.signal})}finally{clearTimeout(timer)}}
async function initialize(){try{const config=await fetch(`./data/api-config.json?ts=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),url=new URL(config.apiBase);if(url.protocol!=='https:'||url.pathname!=='/'||url.username||url.password)throw new Error();apiBase=url.origin;const response=await fetchTimed(`${apiBase}/api/public/health`),health=await response.json();if(!response.ok||health.status!=='ok'||health.data_classification!=='public_notice_demo')throw new Error();live=true;$('#connectionStatus').innerHTML='<i></i> Mac 검색 서버 연결';$('#resultState').textContent='사진을 선택하면 실제 공개 공고를 검색합니다.'}catch{live=false;$('#connectionStatus').innerHTML='<i class="offline"></i> 합성 데모 모드';$('#resultState').textContent='Mac 서버가 꺼져 있어 예시 후보를 표시합니다.'}}
async function searchPhoto(file,fields){
  const form=new FormData();
  form.append('file',file);form.append('top_k',String(SERVER_TOP_K));
  Object.entries(fields).forEach(([key,value])=>form.append(key,value));
  const response=await fetchTimed(`${apiBase}/api/public/search`,{method:'POST',body:form},30000),data=await response.json();
  if(!response.ok)throw new Error(data.detail||'검색 실패');
  return data.items||[];
}

// ---- 이벤트 ----
$('#demoOptions').addEventListener('click',event=>{const option=event.target.closest('.demo-option');if(option)selectDemo(option.dataset.demo)});
$('#photo').addEventListener('change',event=>{selectedDemoId='';addPhotos(event.target.files);event.target.value=''});
$('#photoGrid').addEventListener('click',event=>{const remove=event.target.closest('.photo-remove');if(!remove)return;const [photo]=photos.splice(Number(remove.dataset.index),1);URL.revokeObjectURL(photo.url);selectedDemoId='';renderPhotos()});
['dragover','dragleave','drop'].forEach(type=>$('#photoGrid').addEventListener(type,event=>{event.preventDefault();$('#photoGrid').classList.toggle('dragging',type==='dragover');if(type==='drop'){selectedDemoId='';addPhotos(event.dataTransfer.files)}}));
$('#sido').addEventListener('change',event=>fillSigungu(event.target.value));
$('#sigungu').addEventListener('change',updateDetailPlaceholder);
$('#reroll').addEventListener('click',()=>{const pages=Math.ceil(ranked.length/PAGE_SIZE);page=(page+1)%pages;renderLivePage();$('.results-panel').scrollIntoView({behavior:'smooth',block:'start'})});
button.addEventListener('click',async()=>{
  const breed=$('#breed').value.trim();
  if(!breed){$('#resultState').textContent='강아지 종을 선택해 주세요. 모르면 ‘모름’을 고르면 돼요.';$('#breed').focus();return}
  const fields={feature_text:featureText(),location_text:locationText(),missing_date:$('#date').value,exclude_exact_image:String(Boolean(selectedDemoId))};
  button.disabled=true;
  try{
    if(live){
      if(!photos.length)throw new Error('실제 검색에는 사진이 필요합니다.');
      const lists=[];
      for(const [i,photo] of photos.entries()){button.firstChild.textContent=photos.length>1?`사진 ${i+1}/${photos.length} 검색 중… `:'실제 공고 검색 중… ';lists.push(await searchPhoto(photo.file,fields))}
      ranked=mergeResults(lists);page=0;renderLivePage();
      $('#resultState').textContent=(selectedDemoId?'동일 사진을 제외하고 같은 개체의 다른 사진을 포함한 후보입니다.':'Mac 검색 엔진이 반환한 실제 공개 공고 후보입니다.')+(photos.length>1?` 사진 ${photos.length}장의 결과를 합쳤어요.`:'');
    }else{
      button.firstChild.textContent='합성 후보 정렬 중… ';
      renderStatic(demoCandidates[selectedDemoId]||demoCandidates.shiba);
      $('#resultState').textContent=selectedDemoId?'선택한 합성 예시의 후보 결과입니다.':'Mac 서버가 꺼져 있어 사용자 사진 대신 합성 예시 후보를 표시합니다.';
    }
  }catch(error){$('#resultState').textContent=error.message||'검색 서버와 통신하지 못했습니다.'}
  finally{button.disabled=false;button.firstChild.textContent='유사 후보 살펴보기 ';$('.results-panel').scrollIntoView({behavior:'smooth',block:'start'})}
});
async function startDemo(){renderPhotos();renderDemos();renderStatic(demoCandidates.shiba);await Promise.all([initialize(),loadOptions()]);await selectDemo('shiba')}
startDemo();
