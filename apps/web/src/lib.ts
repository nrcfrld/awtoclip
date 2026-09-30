import {clsx, type ClassValue} from 'clsx'
import {twMerge} from 'tailwind-merge'
export const cn = (...inputs:ClassValue[]) => twMerge(clsx(inputs))
export const stamp = (seconds:number) => {const s=Math.floor(seconds||0);return `${Math.floor(s/3600)>0?`${Math.floor(s/3600)}:`:''}${String(Math.floor(s/60)%60).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`}
export async function api<T>(path:string, options?:RequestInit):Promise<T>{const response=await fetch(`/api${path}`,options);if(!response.ok){let message=`Request failed (${response.status})`;try{const body=await response.json();message=typeof body.detail==='string'?body.detail:message}catch{}throw new Error(message)}return response.json() as Promise<T>}
export const json=(method:string, body:unknown):RequestInit=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})

