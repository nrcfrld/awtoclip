import {useEffect,useMemo,useRef,useState} from 'react'
import {useQuery,useQueryClient} from '@tanstack/react-query'
import {ArrowDownToLine,Check,ChevronDown,Clock3,FileVideo2,LoaderCircle,Pause,Play,Plus,Search,Scissors,Settings2,Sparkles,UploadCloud,X} from 'lucide-react'
import {api,json,stamp} from './lib'
import {Button} from './ui/button'

type Stage='pending'|'running'|'completed'|'failed'|'paused'
type TranscriptionProgress={processed_audio_seconds:number;total_audio_seconds:number;progress_percentage:number;realtime_speed:number|null;estimated_remaining_seconds:number|null;eta_message:string|null;status:string}
type Project={id:string;created_at:string;original_filename:string|null;duration:number|null;max_clip_duration:number;state:string;progress:number|null;error:string|null;stages:Record<string,Stage>;width?:number;height?:number;fps?:number;codec?:string;file_size?:number;transcription_progress?:TranscriptionProgress;analysis_progress?:{completed_chunks:number;paused:boolean}}
type Segment={start:number;end:number;text:string;words?:unknown[]}
type Moment={id:string;start:number;end:number;title:string;category:string;score:number;reason:string;excerpt:string;selected:boolean;rejected:boolean;source:string}
type Clip={filename:string;url:string}
const categories=['all','educational','question','breakthrough','testimonial','promotional','funny','other']
const steps=[['upload','Video upload'],['audio','Audio extraction'],['transcription','Transcription'],['global_context','Session context'],['analysis','AI analysis']]

function App(){
  const client=useQueryClient()
  const [projectId,setProjectId]=useState(new URLSearchParams(location.search).get('project'))
  const [file,setFile]=useState<File|null>(null)
  const [maxDuration,setMaxDuration]=useState(60)
  const [busy,setBusy]=useState(false)
  const [message,setMessage]=useState('')
  const [category,setCategory]=useState('all')
  const [search,setSearch]=useState('')
  const [now,setNow]=useState(0)
  const [previewEnd,setPreviewEnd]=useState<number|null>(null)
  const [manualStart,setManualStart]=useState<number|null>(null)
  const [manualEnd,setManualEnd]=useState<number|null>(null)
  const [manualTitle,setManualTitle]=useState('')
  const [editing,setEditing]=useState<string|null>(null)
  const [editStart,setEditStart]=useState(0)
  const [editEnd,setEditEnd]=useState(0)
  const [clips,setClips]=useState<Clip[]>([])
  const player=useRef<HTMLVideoElement>(null)
  const transcriptRef=useRef<HTMLDivElement>(null)
  const projects=useQuery({queryKey:['projects'],queryFn:()=>api<Project[]>('/projects')})
  const detail=useQuery({queryKey:['project',projectId],queryFn:()=>api<Project>(`/projects/${projectId}`),enabled:!!projectId})
  const transcript=useQuery({queryKey:['transcript',projectId],queryFn:()=>api<{segments:Segment[]}>(`/projects/${projectId}/transcript`),enabled:!!projectId,retry:false})
  const moments=useQuery({queryKey:['moments',projectId],queryFn:()=>api<{moments:Moment[]}>(`/projects/${projectId}/moments`),enabled:!!projectId,retry:false})
  const current=detail.data
  const visible=useMemo(()=>[...(moments.data?.moments||[])].filter(m=>!m.rejected&&(category==='all'||m.category===category)).sort((a,b)=>b.score-a.score),[moments.data,category])
  const selected=(moments.data?.moments||[]).filter(m=>m.selected&&!m.rejected)

  useEffect(()=>{if(!projectId)return;const stream=new EventSource(`/api/projects/${projectId}/events`);stream.onmessage=event=>{const updated=JSON.parse(event.data) as Project;client.setQueryData(['project',projectId],updated);if(updated.state==='READY'||updated.state==='FAILED'){client.invalidateQueries({queryKey:['transcript',projectId]});client.invalidateQueries({queryKey:['moments',projectId]});client.invalidateQueries({queryKey:['projects']})}};stream.addEventListener('transcript_segment',event=>{const segment=JSON.parse((event as MessageEvent).data) as Segment;client.setQueryData<{segments:Segment[]}>(['transcript',projectId],previous=>{const existing=previous?.segments||[];return existing.some(item=>item.start===segment.start&&item.end===segment.end&&item.text===segment.text)?{segments:existing}:{segments:[...existing,segment]}})});return()=>stream.close()},[projectId,client])
  useEffect(()=>{const url=new URL(location.href);if(projectId)url.searchParams.set('project',projectId);else url.searchParams.delete('project');history.replaceState(null,'',url)},[projectId])
  useEffect(()=>{if(previewEnd!==null&&now>=previewEnd){player.current?.pause();setPreviewEnd(null)}},[now,previewEnd])

  const showError=(error:unknown)=>setMessage(error instanceof Error?error.message:String(error))
  async function startProject(){if(!file)return;setBusy(true);setMessage('');try{const created=await api<Project>('/projects',json('POST',{max_clip_duration:maxDuration}));setProjectId(created.id);const upload=new FormData();upload.append('video',file);await api<Project>(`/projects/${created.id}/video`,{method:'POST',body:upload});client.invalidateQueries({queryKey:['projects']});await api(`/projects/${created.id}/process`,{method:'POST'});setFile(null)}catch(error){showError(error)}finally{setBusy(false)}}
  async function retryUpload(){if(!projectId||!file)return;setBusy(true);setMessage('');try{const upload=new FormData();upload.append('video',file);await api<Project>(`/projects/${projectId}/video`,{method:'POST',body:upload});client.invalidateQueries({queryKey:['projects']});await api(`/projects/${projectId}/process`,{method:'POST'});setFile(null)}catch(error){showError(error)}finally{setBusy(false)}}
  async function process(){if(!projectId)return;setMessage('');try{await api(`/projects/${projectId}/process`,{method:'POST'})}catch(error){showError(error)}}
  function seek(time:number,play=false){if(!player.current)return;player.current.currentTime=time;setNow(time);if(play)player.current.play().catch(()=>{})}
  function preview(moment:Moment){setPreviewEnd(moment.end);seek(moment.start,true)}
  async function patch(moment:Moment,changes:Partial<Moment>){if(!projectId)return;try{await api(`/projects/${projectId}/moments/${moment.id}`,json('PATCH',changes));client.invalidateQueries({queryKey:['moments',projectId]});setEditing(null)}catch(error){showError(error)}}
  async function createManual(){if(!projectId||manualStart===null||manualEnd===null)return;try{const start=Math.min(manualStart,manualEnd),end=Math.max(manualStart,manualEnd);await api(`/projects/${projectId}/moments`,json('POST',{start,end,title:manualTitle||'Manual clip',category:'other'}));client.invalidateQueries({queryKey:['moments',projectId]});setManualStart(null);setManualEnd(null);setManualTitle('')}catch(error){showError(error)}}
  async function exportClips(ids?:string[]){if(!projectId)return;setBusy(true);setMessage('');try{const result=await api<{clips:Clip[]}>(`/projects/${projectId}/clips`,json('POST',{moment_ids:ids}));setClips(result.clips);client.invalidateQueries({queryKey:['project',projectId]})}catch(error){showError(error)}finally{setBusy(false)}}
  const isWorking=current&&['UPLOADING','AUDIO_EXTRACTING','TRANSCRIBING','BUILDING_CONTEXT','ANALYZING','REFINING','EXPORTING'].includes(current.state)
  const transcriptionRunning=current?.transcription_progress?.status==='running'||current?.transcription_progress?.status==='loading_model'
  const displayedProgress=transcriptionRunning?current?.transcription_progress?.progress_percentage:current?.progress

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Scissors size={21} strokeWidth={2.5}/></span><span>awto<span className="brand-light">clip</span></span></div>
      <div className="sidebar-caption">YOUR WORKSPACE</div>
      <button className={`nav-item ${!projectId?'active':''}`} onClick={()=>{setProjectId(null);setMessage('')}}><Plus size={18}/> New project</button>
      <div className="sidebar-caption mt-8">RECENT PROJECTS</div>
      <div className="project-list">{projects.data?.map(p=><button key={p.id} className={`project-item ${projectId===p.id?'selected':''}`} onClick={()=>{setProjectId(p.id);setMessage('')}}><FileVideo2 size={16}/><span className="truncate">{p.original_filename||'Untitled project'}</span></button>)}</div>
      <div className="sidebar-footer"><span className="status-dot"/> Local workspace <span className="ml-auto">v0.1</span></div>
    </aside>

    <main className="main-panel">
      <header className="topbar"><div className="breadcrumb"><span>Workspace</span><span className="slash">/</span><strong>{current?.original_filename||'New project'}</strong></div><span className="privacy"><span className="status-dot"/> Your media stays local</span></header>
      {message&&<div className="alert" role="alert"><span>{message}</span><button onClick={()=>setMessage('')} aria-label="Dismiss error"><X size={17}/></button></div>}

      {!projectId?<div className="new-page">
        <div className="eyebrow"><span className="tiny-line"/> TURN LONG VIDEOS INTO SHORT STORIES</div>
        <h1>Find the moments<br/><em>worth sharing.</em></h1>
        <p className="hero-copy">Your best clips are already in the recording. Upload a video, let AI surface the standout moments, then make the final cut your own.</p>
        <div className="upload-card">
          <label className={`dropzone ${file?'has-file':''}`} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();const dropped=e.dataTransfer.files[0];if(dropped)setFile(dropped)}}>
            <input type="file" accept=".mp4,.mov,.mkv,video/mp4,video/quicktime,video/x-matroska" onChange={e=>setFile(e.target.files?.[0]||null)}/>
            <span className="upload-icon"><UploadCloud size={28}/></span>
            <strong>{file?file.name:'Drop your recording here'}</strong>
            <span>{file?`${(file.size/1024/1024).toFixed(1)} MB · Ready to upload`:'or click to browse your files'}</span>
            <small>MP4, MOV or MKV · Recordings up to around 2 hours</small>
          </label>
          <div className="upload-bottom"><label className="duration-label" htmlFor="max-duration"><Settings2 size={17}/> Maximum clip length</label><div className="duration-control"><input id="max-duration" type="number" min="5" max="600" value={maxDuration} onChange={e=>setMaxDuration(Number(e.target.value))}/><span>seconds</span></div><Button variant="accent" size="lg" disabled={!file||busy} onClick={startProject}>{busy?<LoaderCircle className="spin" size={17}/>:<Sparkles size={17}/>} {busy?'Uploading…':'Find moments'}</Button></div>
        </div>
        <div className="feature-row"><div><span>01</span><strong>Upload locally</strong><p>Your video and audio stay on your machine.</p></div><div><span>02</span><strong>Discover moments</strong><p>AI scans the transcript for standalone ideas.</p></div><div><span>03</span><strong>Make the cut</strong><p>Review, fine tune and export only what you choose.</p></div></div>
      </div>:<div className="workspace">
        <div className="workspace-heading"><div><div className="eyebrow"><span className="tiny-line"/> PROJECT STUDIO</div><h1>{current?.original_filename||'New project'}</h1><div className="meta-line">{current?.duration?stamp(current.duration):'Preparing video'}<span>·</span>{current?.width&&current?.height?`${current.width} × ${current.height}`:'—'}<span>·</span>{current?.codec?.toUpperCase()||'—'}</div></div><div className="heading-actions"><span className={`stage-pill ${current?.state==='READY'?'ready':''}`}>{isWorking?<LoaderCircle size={14} className="spin"/>:<span className="status-dot"/>}{current?.state?.replace(/_/g,' ')||'LOADING'}</span><Button variant="accent" disabled={!selected.length||busy} onClick={()=>exportClips()}><ArrowDownToLine size={16}/> Export selected <span className="count-bubble">{selected.length}</span></Button></div></div>

        {(current?.state==='FAILED'||current?.state==='ANALYSIS_PAUSED')&&<div className="failure"><strong>{current.state==='ANALYSIS_PAUSED'?'AI analysis paused':'Processing stopped'}</strong><span>{current.error}</span>{current.original_filename&&<Button size="sm" onClick={process}>Retry processing</Button>}</div>}
        {current&&!current.original_filename&&<div className="retry-upload"><label>Choose a source video to finish this project<input type="file" accept=".mp4,.mov,.mkv" onChange={e=>setFile(e.target.files?.[0]||null)}/></label><Button disabled={!file||busy} onClick={retryUpload}>{busy?'Uploading…':'Upload and process'}</Button></div>}
        {current&&current.state!=='READY'&&current.state!=='FAILED'&&current.state!=='ANALYSIS_PAUSED'&&<div className="progress-card"><div className="progress-header"><div><Sparkles size={18}/><strong>Finding your best moments</strong></div><span>{displayedProgress!==null&&displayedProgress!==undefined?`${Math.round(displayedProgress)}%`:'Starting'}</span></div><div className="progress-track"><div style={{transform:`scaleX(${Math.min(100,Math.max(0,displayedProgress||0))/100})`}}/></div>{transcriptionRunning&&current.transcription_progress&&<div className="progress-details"><span>{stamp(current.transcription_progress.processed_audio_seconds)} / {stamp(current.transcription_progress.total_audio_seconds)} transcribed</span><span>{current.transcription_progress.realtime_speed?.toFixed(1)||'—'}× realtime</span><span>{current.transcription_progress.estimated_remaining_seconds!==null?`About ${stamp(current.transcription_progress.estimated_remaining_seconds)} remaining`:current.transcription_progress.eta_message||'Estimating remaining time...'}</span></div>}{current.analysis_progress&&<div className="progress-details"><span>{current.analysis_progress.completed_chunks} analysis chunks complete</span>{current.analysis_progress.paused&&<span>Analysis paused; transcription continues</span>}</div>}<div className="stage-row">{steps.map(([key,label])=><div key={key} className={current.stages[key]==='completed'?'complete':''}><span className="stage-icon">{current.stages[key]==='completed'?<Check size={13}/>:current.stages[key]==='running'?<LoaderCircle size={13} className="spin"/>:<span/>}</span>{label}</div>)}</div></div>}

        <div className="studio-grid"><section className="player-column"><div className="video-frame">{current?.original_filename?<video ref={player} src={`/api/projects/${projectId}/video`} controls preload="metadata" onTimeUpdate={e=>setNow(e.currentTarget.currentTime)}/>:<div className="video-placeholder"><FileVideo2 size={38}/><span>Waiting for video upload</span></div>}</div><div className="under-player"><div><span className="playhead-tag">PLAYHEAD</span><strong>{stamp(now)}</strong><span className="muted">/ {stamp(current?.duration||0)}</span></div>{previewEnd!==null?<Button size="sm" variant="ghost" onClick={()=>setPreviewEnd(null)}><Pause size={14}/> Free playback</Button>:<span className="preview-hint">Preview directly from your original video</span>}</div>
          <div className="transcript-panel"><div className="panel-head"><div><h2>Transcript</h2><span>{transcript.data?.segments.length||0} segments</span></div><div className="search-box"><Search size={15}/><input aria-label="Search transcript" value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search transcript"/></div></div><div className="transcript-list" ref={transcriptRef}>{transcript.data?.segments.filter(s=>!search||s.text.toLowerCase().includes(search.toLowerCase())).map((segment,index)=>{const covered=(moments.data?.moments||[]).some(m=>!m.rejected&&segment.end>m.start&&segment.start<m.end);return <button key={`${segment.start}-${index}`} className={`transcript-line ${segment.start<=now&&segment.end>=now?'playing':''} ${covered?'covered':''}`} onClick={()=>seek(segment.start,true)}><span>{stamp(segment.start)}</span><p>{segment.text}</p>{covered&&<i/>}</button>})||<div className="empty-small">Transcript appears after transcription.</div>}</div></div>
          <div className="manual-panel"><div className="manual-title"><Scissors size={18}/><div><strong>Create a clip manually</strong><span>Mark an exact range from the playhead</span></div></div><div className="manual-controls"><Button size="sm" variant="outline" onClick={()=>setManualStart(now)}>Set start <span>{manualStart!==null?stamp(manualStart):'—'}</span></Button><Button size="sm" variant="outline" onClick={()=>setManualEnd(now)}>Set end <span>{manualEnd!==null?stamp(manualEnd):'—'}</span></Button><Button size="sm" variant="ghost" disabled={manualStart===null||manualEnd===null} onClick={()=>{if(player.current){setPreviewEnd(Math.max(manualStart!,manualEnd!));seek(Math.min(manualStart!,manualEnd!),true)}}}><Play size={14}/> Preview</Button></div><div className="manual-save"><input aria-label="Manual clip title" placeholder="Name this moment" value={manualTitle} onChange={e=>setManualTitle(e.target.value)}/><Button size="sm" disabled={manualStart===null||manualEnd===null} onClick={createManual}>Save candidate</Button></div></div>
        </section>

        <section className="moments-column"><div className="moments-head"><div><div className="eyebrow">DISCOVERED IN YOUR VIDEO</div><h2>{(moments.data?.moments||[]).filter(m=>!m.rejected).length} Moments Found</h2></div><span className="sort-label">Sorted by score <ChevronDown size={14}/></span></div><div className="filters">{categories.map(c=><button key={c} className={category===c?'active':''} onClick={()=>setCategory(c)}>{c==='all'?'All moments':c}</button>)}</div><div className="moment-list">{visible.length?visible.map((moment,index)=><article key={moment.id} className={`moment-card ${moment.selected?'is-selected':''}`}><div className="card-top"><span className="moment-number">{String(index+1).padStart(2,'0')}</span><span className="category-tag">{moment.category}</span><span className="score"><Sparkles size={13}/>{moment.score}</span></div><h3>{moment.title}</h3><div className="time-line"><Clock3 size={14}/>{stamp(moment.start)} <span>→</span> {stamp(moment.end)} <span className="duration">{Math.round(moment.end-moment.start)}s</span></div><p className="excerpt">“{moment.excerpt||'No transcript excerpt available.'}”</p><p className="reason">{moment.reason}</p>{editing===moment.id&&<div className="edit-row"><label>Start <input type="number" min="0" step="0.1" value={editStart} onChange={e=>setEditStart(Number(e.target.value))}/></label><label>End <input type="number" min="0" step="0.1" value={editEnd} onChange={e=>setEditEnd(Number(e.target.value))}/></label><Button size="sm" onClick={()=>patch(moment,{start:editStart,end:editEnd})}>Save</Button></div>}<div className="card-actions"><Button size="sm" variant="ghost" onClick={()=>preview(moment)}><Play size={14}/> Preview</Button><Button size="sm" variant="ghost" onClick={()=>{setEditing(editing===moment.id?null:moment.id);setEditStart(moment.start);setEditEnd(moment.end)}}><Settings2 size={14}/> Edit</Button><Button size="sm" variant={moment.selected?'accent':'outline'} onClick={()=>patch(moment,{selected:!moment.selected})}>{moment.selected?<Check size={14}/>:<Plus size={14}/>} {moment.selected?'Selected':'Select'}</Button><button className="reject" aria-label={`Reject ${moment.title}`} title="Reject moment" onClick={()=>patch(moment,{rejected:true})}><X size={16}/></button></div><button className="single-export" onClick={()=>exportClips([moment.id])} disabled={busy||!moment.selected}><ArrowDownToLine size={13}/> Generate clip</button></article>):<div className="no-moments"><Sparkles size={28}/><strong>{current?.state==='READY'?'No moments in this category yet':'Moments will appear here'}</strong><p>{current?.state==='READY'?'Try another filter or create a clip manually.':'We’ll show candidates once analysis is complete.'}</p></div>}</div>{clips.length>0&&<div className="export-panel"><strong>Exported clips</strong>{clips.map(clip=><a key={clip.filename} href={`/api${clip.url}`} download><ArrowDownToLine size={14}/>{clip.filename}</a>)}</div>}</section></div>
      </div>}
    </main>
  </div>
}
export default App
