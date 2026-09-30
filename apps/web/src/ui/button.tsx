import * as React from 'react'
import {Slot} from '@radix-ui/react-slot'
import {cva, type VariantProps} from 'class-variance-authority'
import {cn} from '../lib'
const variants=cva('inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-moss focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',{variants:{variant:{default:'bg-ink text-white hover:bg-moss',accent:'bg-lime text-ink hover:bg-[#cce85a]',outline:'border border-line bg-white hover:bg-paper',ghost:'hover:bg-[#e8ebe3] text-ink',danger:'bg-[#fce7e4] text-[#a54334] hover:bg-[#f6d4cf]'},size:{default:'h-10 px-4',sm:'h-8 px-3 text-xs',lg:'h-12 px-6'}},defaultVariants:{variant:'default',size:'default'}})
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>,VariantProps<typeof variants>{asChild?:boolean}
export const Button=React.forwardRef<HTMLButtonElement,ButtonProps>(({className,variant,size,asChild=false,...props},ref)=>{const Component=asChild?Slot:'button';return <Component className={cn(variants({variant,size}),className)} ref={ref} {...props}/>})
Button.displayName='Button'
